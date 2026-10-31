"""Evaluering / Evaluate the trained brain: accuracy per category, confusion matrix.

  python evaluate.py                          # samme val-split som under trening / same val split as training
  python evaluate.py --data data/eval_real.csv   # ANBEFALT: ekte tekst modellen aldri har sett / recommended:
                                                 # real text the model has never seen
  python evaluate.py --data mine.csv --show-errors
"""

import argparse
import sys
from typing import Dict, List, Sequence, Tuple

from data import load_records, make_targets, split_records
from predict import Brain
from utils import say


def evaluate(brain: Brain, records: Sequence[Tuple[str, str]]) -> dict:
    """Returnerer accuracy, per-klasse precision/recall/F1 og confusion matrix.
    Labels som modellen ikke kjenner telles som feil. / Unknown labels count as wrong."""
    classes: List[str] = list(brain.classes)
    texts = [t for t, _ in records]
    preds = [r["label"] for r in brain.predict_batch(texts)]
    gold = [l for _, l in records]
    labels = classes + sorted({g for g in gold if g not in classes})
    conf: Dict[str, Dict[str, int]] = {g: {p: 0 for p in classes} for g in labels}
    for g, p in zip(gold, preds):
        conf[g][p] += 1
    per = {}
    for c in labels:
        tp = conf[c].get(c, 0)
        support = sum(conf[c].values())
        predicted = sum(conf[g].get(c, 0) for g in labels)
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    acc = sum(g == p for g, p in zip(gold, preds)) / len(gold)
    return {"accuracy": acc, "per_class": per, "confusion": conf, "classes": classes, "preds": preds, "gold": gold}


def report(res: dict, texts: Sequence[str], show_errors: bool = False) -> None:
    print(f"\nNøyaktighet / Accuracy: {res['accuracy']:.1%}  ({len(res['gold'])} eksempler/examples)\n")
    print(f"{'Kategori / Category':<24}{'Prec':>7}{'Recall':>8}{'F1':>7}{'N':>6}")
    for c, m in res["per_class"].items():
        print(f"{c:<24}{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}{m['support']:>6d}")
    print("\nForvirringsmatrise / Confusion matrix (rader=fasit/rows=truth, kolonner=prediksjon/cols=predicted):")
    short = [c[:9] for c in res["classes"]]
    print(f"{'':<24}" + "".join(f"{s:>10}" for s in short))
    for g, row in res["confusion"].items():
        print(f"{g:<24}" + "".join(f"{row[p]:>10d}" for p in res["classes"]))
    if show_errors:
        print("\nFeil / Errors:")
        for t, g, p in zip(texts, res["gold"], res["preds"]):
            if g != p:
                print(f"  [{g} -> {p}] {t[:100]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluer hjernen / Evaluate the brain")
    ap.add_argument("--model", default="models/brain.pth")
    ap.add_argument("--data", help="Egen labelt CSV/JSON. Standard: val-split fra treningen / default: training val split")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--show-errors", action="store_true")
    a = ap.parse_args(argv)
    try:
        brain = Brain(a.model)
        if brain.task != "classification":
            say("Evaluering er kun for klassifisering.", "Evaluation is for classification models only.")
            return 1
        if a.data:
            records = load_records(a.data, a.text_col, a.label_col)
            say(f"Evaluerer på {a.data}", f"Evaluating on {a.data}")
        else:
            ck = brain.checkpoint
            allrec = load_records(ck["data"], ck.get("text_col", "text"), ck.get("label_col", "label"))
            targets, _ = make_targets(allrec, "classification")
            _, val_idx = split_records(allrec, targets, ck["val_split"], ck["seed"], "classification")
            records = [allrec[i] for i in val_idx]
            say(f"Evaluerer på val-splitten fra treningen ({len(records)} eksempler). Merk: den er optimistisk hvis "
                f"treningsdata er malgenerert - bruk --data data/eval_real.csv.",
                f"Evaluating on the training val split ({len(records)} examples). Note: optimistic if the training data "
                f"is template-generated - prefer --data data/eval_real.csv.")
        res = evaluate(brain, records)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        say(f"FEIL: {e}", f"ERROR: {e}")
        return 1
    report(res, [t for t, _ in records], a.show_errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
