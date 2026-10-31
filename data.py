"""Datahåndtering / Data loading and preparation.

Supports CSV (columns: text,label) and JSON (list of {"text":..., "label":...}).
Støtter CSV og JSON. Bruk add_example() for å legge til egne eksempler.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str, ngrams: bool = False) -> List[str]:
    """Små bokstaver, ord + tegn-n-gram (som fastText) slik at beslektede ord og norske bøyninger
    deler tokens (monetize/monetization). / Lowercase words plus character 3-5-grams (fastText-style)."""
    words = _TOKEN_RE.findall(str(text).lower())
    if not ngrams:
        return words
    toks = list(words)
    for w in words:
        pw = f"<{w}>"
        for n in (3, 4, 5):
            toks += [pw[i:i + n] for i in range(len(pw) - n + 1)] if len(pw) > n else []
    return toks


class Vocab:
    """Ord -> id. Index 0 = padding, 1 = unknown."""

    def __init__(self, itos: Optional[List[str]] = None):
        self.itos = itos or [PAD, UNK]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Sequence[str], min_freq: int = 1, max_size: Optional[int] = None) -> "Vocab":
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:max_len]
        return ids + [0] * (max_len - len(ids))

    def __len__(self) -> int:
        return len(self.itos)


# ---------------------------------------------------------------- loading / saving

def load_records(path: str, text_col: str = "text", label_col: str = "label") -> List[Tuple[str, str]]:
    """Les (tekst, label) fra CSV eller JSON. / Read (text, label) pairs from CSV or JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fant ikke datafil / Data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(rows, dict):  # allow {"data": [...]}
            rows = rows.get("data", [])
    else:
        raise ValueError(f"Bruk .csv eller .json / Use .csv or .json (got '{suffix}')")
    records = []
    for i, r in enumerate(rows):
        if text_col not in r or label_col not in r:
            raise KeyError(f"Rad/row {i}: mangler kolonne / missing column '{text_col}' or '{label_col}'")
        text, label = str(r[text_col]).strip(), str(r[label_col]).strip()
        if text and label:
            records.append((text, label))
    if not records:
        raise ValueError("Ingen gyldige rader / No valid rows found")
    return records


def add_example(path: str, text: str, label: str, text_col: str = "text", label_col: str = "label") -> None:
    """Legg til ett nytt merket eksempel i datafilen. / Append one labeled example to the data file."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        new = not p.exists() or p.stat().st_size == 0
        with p.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[text_col, label_col])
            if new:
                w.writeheader()
            w.writerow({text_col: text, label_col: label})
    elif p.suffix.lower() == ".json":
        rows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        rows.append({text_col: text, label_col: label})
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError("Bruk .csv eller .json / Use .csv or .json")


# ---------------------------------------------------------------- torch dataset

class TextDataset(Dataset):
    def __init__(self, records: Sequence[Tuple[str, str]], vocab: Vocab, targets: List,
                 max_len: int, task: str):
        self.x = torch.tensor([vocab.encode(t, max_len) for t, _ in records], dtype=torch.long)
        dtype = torch.long if task == "classification" else torch.float32
        self.y = torch.tensor(targets, dtype=dtype)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def make_targets(records, task: str) -> Tuple[List, List[str]]:
    """Klasse-indekser (classification) eller floats (regression)."""
    if task == "classification":
        classes = sorted({l for _, l in records})
        idx = {c: i for i, c in enumerate(classes)}
        return [idx[l] for _, l in records], classes
    try:
        return [float(l) for _, l in records], []
    except ValueError as e:
        raise ValueError("Regresjon krever numeriske labels / Regression needs numeric labels") from e


def split_records(records, targets, val_split: float, seed: int, task: str):
    """Train/val-split (stratifisert for klassifisering). / Stratified for classification."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(records)))
    if task == "classification":
        by_class: Dict[int, List[int]] = {}
        for i in idx:
            by_class.setdefault(targets[i], []).append(i)
        val_idx = []
        for items in by_class.values():
            rng.shuffle(items)
            val_idx += items[:max(1, round(len(items) * val_split))] if len(items) > 1 else []
    else:
        rng.shuffle(idx)
        val_idx = idx[:max(1, round(len(idx) * val_split))]
    val_set = set(val_idx)
    train_idx = [i for i in idx if i not in val_set]
    return train_idx, val_idx


def make_loaders(records, task: str = "classification", val_split: float = 0.2, batch_size: int = 16,
                 max_len: int = 64, min_freq: int = 1, seed: int = 42):
    """Returnerer train_loader, val_loader, vocab, classes. Vocab bygges kun fra treningsdata."""
    targets, classes = make_targets(records, task)
    train_idx, val_idx = split_records(records, targets, val_split, seed, task)
    if len(train_idx) < 2 or not val_idx:
        raise ValueError("For lite data til train/val-split / Too little data for a train/val split")
    vocab = Vocab.build([records[i][0] for i in train_idx], min_freq=min_freq)
    mk = lambda ids: TextDataset([records[i] for i in ids], vocab, [targets[i] for i in ids], max_len, task)
    train_ds, val_ds = mk(train_idx), mk(val_idx)
    # BatchNorm krasjer på batch med 1 eksempel -> dropp siste ufullstendige batch om nødvendig
    drop_last = len(train_ds) % batch_size == 1
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, generator=g)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 32))
    return train_loader, val_loader, vocab, classes


if __name__ == "__main__":
    # python data.py add "my text" "Blues/Roots" [--file data/example_data.csv]
    import argparse
    ap = argparse.ArgumentParser(description="Legg til eksempel / Add a labeled example")
    ap.add_argument("action", choices=["add"])
    ap.add_argument("text")
    ap.add_argument("label")
    ap.add_argument("--file", default="data/example_data.csv")
    ns = ap.parse_args()
    add_example(ns.file, ns.text, ns.label)
    print(f"Lagt til / Added to {ns.file}: [{ns.label}] {ns.text}")
