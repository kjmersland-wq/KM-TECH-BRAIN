"""Trening / Training loop.

Kjør / Run:  python train.py
Egen data / Own data:  python train.py --data mydata.csv --epochs 300
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import load_records, make_loaders
from embeddings import DEFAULT_MODEL, load_encoder
from model import build_model
from utils import get_device, say, set_seed


def run_epoch(model, loader, loss_fn, device, task, optimizer=None):
    """En epoke. Med optimizer = trening, uten = validering. Returnerer (loss, accuracy|None)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if task == "regression":
                out = out.squeeze(-1)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            if task == "classification":
                correct += (out.argmax(1) == y).sum().item()
    return total_loss / n, (correct / n if task == "classification" else None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tren din lokale hjerne / Train your local brain")
    p.add_argument("--data", default="data/example_data.csv", help="CSV eller JSON / CSV or JSON file")
    p.add_argument("--out", default="models/brain.pth", help="Hvor modellen lagres / output .pth path")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128], help="Størrelse på skjulte lag / hidden sizes")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no-batchnorm", action="store_true", help="Slå av BatchNorm / disable BatchNorm")
    p.add_argument("--patience", type=int, default=40, help="Early stopping-tålmodighet / patience (epochs)")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--mode", choices=["simple", "pretrained", "hashed"], default="simple",
                   help="simple = egen embedding (standard) | hashed = lettvekts | pretrained = sentence-transformers")
    p.add_argument("--encoder-model", default=DEFAULT_MODEL,
                   help="Modellnavn eller lokal mappe for pretrained / model name or local folder")
    p.add_argument("--download", action="store_true",
                   help="Tillat nedlasting av pretrained-modellen (kun første gang) / allow one-time model download")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    set_seed(a.seed)
    device = get_device()

    encoder, mode = load_encoder(a.mode, a.encoder_model, a.download)  # faller tilbake til simple / may fall back
    try:
        records = load_records(a.data, a.text_col, a.label_col)
        train_loader, val_loader, vocab, classes = make_loaders(
            records, a.task, a.val_split, a.batch_size, a.max_len, a.min_freq, a.seed, encoder)
    except (FileNotFoundError, KeyError, ValueError) as e:
        say(f"FEIL ved lasting av data: {e}", f"ERROR loading data: {e}")
        return 1

    vocab_txt = f"vokabular {len(vocab)}" if vocab else f"vektorer {encoder.dim}d"
    say(f"Data: {len(records)} eksempler, {len(classes)} klasser, modus {mode}, {vocab_txt}, enhet: {device}",
        f"Data: {len(records)} examples, {len(classes)} classes, mode {mode}, "
        f"{'vocab ' + str(len(vocab)) if vocab else 'vectors ' + str(encoder.dim) + 'd'}, device: {device}")

    n_out = len(classes) if a.task == "classification" else 1
    if encoder is None:
        config = dict(kind="text", vocab_size=len(vocab), num_outputs=n_out, embed_dim=a.embed_dim,
                      hidden_sizes=a.hidden, dropout=a.dropout, use_batchnorm=not a.no_batchnorm, task=a.task)
    else:
        config = dict(kind="dense", input_size=encoder.dim, num_outputs=n_out, hidden_sizes=a.hidden,
                      dropout=a.dropout, use_batchnorm=not a.no_batchnorm, task=a.task)
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"Modell opprettet ({n_params:,} parametere)", f"Model created ({n_params:,} parameters)")

    loss_fn = nn.CrossEntropyLoss() if a.task == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(3, a.patience // 4))

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, a.task, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device, a.task)
        scheduler.step(va_loss)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, train_acc=tr_acc, val_acc=va_acc))

        if va_loss < best_loss - 1e-4:  # beste modell så langt -> lagre / new best -> keep
            best_loss, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1

        if epoch == 1 or epoch % a.log_every == 0:
            acc = f" | acc {tr_acc:.3f}/{va_acc:.3f}" if tr_acc is not None else ""
            print(f"  epoke/epoch {epoch:4d} | loss {tr_loss:.4f}/{va_loss:.4f}{acc} | lr {optimizer.param_groups[0]['lr']:.2e}")
        if bad >= a.patience:
            say(f"Early stopping etter epoke {epoch} (ingen forbedring på {a.patience} epoker)",
                f"Early stopping at epoch {epoch} (no improvement for {a.patience} epochs)")
            break

    model.load_state_dict(best_state)
    best = history[best_epoch - 1]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {k: v.cpu() for k, v in best_state.items()}, "config": config,
                "vocab": vocab.itos if vocab else None, "classes": classes, "max_len": a.max_len,
                "mode": mode, "encoder": encoder.spec() if encoder else None,
                "seed": a.seed, "val_split": a.val_split, "data": a.data,
                "text_col": a.text_col, "label_col": a.label_col,
                "best_epoch": best_epoch, "history": history}, out)

    acc_txt = f", val-nøyaktighet/accuracy {best['val_acc']:.3f}" if best["val_acc"] is not None else ""
    say(f"Ferdig på {time.time() - start:.1f}s. Beste epoke {best_epoch}, val-loss {best_loss:.4f}{acc_txt}. Lagret: {out}",
        f"Done in {time.time() - start:.1f}s. Best epoch {best_epoch}, val loss {best_loss:.4f}{acc_txt}. Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
