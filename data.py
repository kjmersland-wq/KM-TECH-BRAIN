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
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
                 max_len: int = 64, min_freq: int = 1, seed: int = 42, encoder=None):
    """Returnerer train_loader, val_loader, vocab, classes. Vocab bygges kun fra treningsdata.
    Med `encoder` (hashed/pretrained) brukes ferdige vektorer og vocab er None.
    With an `encoder` the texts become dense vectors and vocab is None."""
    targets, classes = make_targets(records, task)
    train_idx, val_idx = split_records(records, targets, val_split, seed, task)
    if len(train_idx) < 2 or not val_idx:
        raise ValueError("For lite data til train/val-split / Too little data for a train/val split")
    if encoder is None:
        vocab = Vocab.build([records[i][0] for i in train_idx], min_freq=min_freq)
        mk = lambda ids: TextDataset([records[i] for i in ids], vocab, [targets[i] for i in ids], max_len, task)
    else:
        vocab = None
        feats = encoder.encode([t for t, _ in records])  # encode once / kodes én gang
        ydtype = torch.long if task == "classification" else torch.float32
        mk = lambda ids: TensorDataset(feats[ids], torch.tensor([targets[i] for i in ids], dtype=ydtype))
    train_ds, val_ds = mk(train_idx), mk(val_idx)
    # BatchNorm krasjer på batch med 1 eksempel -> dropp siste ufullstendige batch om nødvendig
    drop_last = len(train_ds) % batch_size == 1
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, generator=g)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 32))
    return train_loader, val_loader, vocab, classes


# ---------------------------------------------------------------- bulk import / stats

TEXT_EXT = {".txt", ".md"}


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _split_text(content: str, unit: str) -> List[str]:
    if unit == "file":
        return [content.strip()]
    if unit == "paragraphs":
        return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    return [l.strip() for l in content.splitlines() if l.strip()]  # lines


def _read_text_file(path: Path, label: Optional[str], unit: str) -> List[Tuple[str, str]]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    out = []
    for chunk in _split_text(content, unit):
        if label:
            out.append((chunk, label))
        elif unit == "lines" and ("\t" in chunk or " | " in chunk):
            lab, txt = chunk.split("\t", 1) if "\t" in chunk else chunk.split(" | ", 1)
            out.append((txt.strip(), lab.strip()))
        else:
            raise ValueError(f"{path}: mangler label / no label. Bruk --label, eller 'label<TAB>tekst' / "
                             f"use --label or 'label<TAB>text' per line")
    return out


def collect_examples(source: str, label: Optional[str] = None, unit: Optional[str] = None,
                     min_chars: int = 3) -> List[Tuple[str, str]]:
    """Samle mange eksempler fra en fil eller mappe. / Gather many examples from a file or a folder.

    * Mappe / folder: undermappenavn = label (f.eks. inbox/Blues_Roots/*.txt), eller --label for alle.
      Ett eksempel per fil (unit=file). Folder name `Blues_Roots` -> label `Blues/Roots` if such a label exists
      in your data; otherwise use the exact label as folder name (use `__` for '/': Blues__Roots).
    * .txt/.md: én tekst per linje (unit=lines) med --label, eller 'label<TAB>tekst' / 'label | tekst'.
      unit=paragraphs: blanke linjer skiller eksempler. unit=file: hele filen er ett eksempel.
    * .csv/.json: leses som vanlig (kolonner text,label). / read as usual.
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Fant ikke / not found: {p}")
    found: List[Tuple[str, str]] = []
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(p)
            lab = label or (rel.parts[0].replace("__", "/") if len(rel.parts) > 1 else None)
            if f.suffix.lower() in (".csv", ".json"):
                found += load_records(str(f))
            elif f.suffix.lower() in TEXT_EXT:
                found += _read_text_file(f, lab, unit or ("file" if lab else "lines"))
    elif p.suffix.lower() in (".csv", ".json"):
        found = load_records(str(p))
    else:
        found = _read_text_file(p, label, unit or "lines")
    return [(t, l) for t, l in found if len(t) >= min_chars and l]


def add_examples(path: str, records: Sequence[Tuple[str, str]], dedupe: bool = True) -> Tuple[int, int]:
    """Legg til mange eksempler på én gang. Returnerer (lagt_til, hoppet_over_duplikater)."""
    existing = set()
    if dedupe and Path(path).exists():
        existing = {_norm(t) for t, _ in load_records(path)}
    added = skipped = 0
    for text, lab in records:
        key = _norm(text)
        if dedupe and key in existing:
            skipped += 1
            continue
        existing.add(key)
        add_example(path, text, lab)
        added += 1
    return added, skipped


def dataset_stats(records: Sequence[Tuple[str, str]]) -> str:
    """Antall per label, ordlengde, duplikater. / Per-label counts, length, duplicates."""
    from collections import Counter
    counts = Counter(l for _, l in records)
    words: Dict[str, int] = Counter()
    for t, l in records:
        words[l] += len(tokenize(t))
    dupes = len(records) - len({_norm(t) for t, _ in records})
    lines = [f"{len(records)} eksempler/examples, {len(counts)} labels, {dupes} duplikater/duplicates"]
    for lab, n in counts.most_common():
        lines.append(f"  {lab:<24} {n:5d}   snitt/avg {words[lab] / n:5.1f} ord/words")
    small = [l for l, n in counts.items() if n < 20]
    if small:
        lines.append(f"  Tips: {', '.join(small)} har under 20 eksempler / have fewer than 20 examples.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Datahjelp / Data tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_add = sub.add_parser("add", help="Ett eksempel / one example")
    a_add.add_argument("text"); a_add.add_argument("label")
    a_imp = sub.add_parser("import", help="Mange eksempler fra fil/mappe / many examples from file or folder")
    a_imp.add_argument("source")
    a_imp.add_argument("--label", help="Label for alle / one label for everything imported")
    a_imp.add_argument("--unit", choices=["lines", "paragraphs", "file"], help="Hva er ett eksempel? / what is one example?")
    a_imp.add_argument("--no-dedupe", action="store_true")
    a_imp.add_argument("--dry-run", action="store_true")
    a_st = sub.add_parser("stats", help="Vis statistikk / show dataset stats")
    for sp in (a_add, a_imp, a_st):
        sp.add_argument("--file", default="data/example_data.csv", help="Datafil / data file (csv or json)")
    ns = ap.parse_args()
    try:
        if ns.cmd == "add":
            add_example(ns.file, ns.text, ns.label)
            print(f"Lagt til / Added to {ns.file}: [{ns.label}] {ns.text}")
        elif ns.cmd == "import":
            recs = collect_examples(ns.source, ns.label, ns.unit)
            if ns.dry_run:
                print(f"{len(recs)} eksempler funnet / found (dry run). Eksempel/sample: {recs[:2]}")
            else:
                added, skipped = add_examples(ns.file, recs, dedupe=not ns.no_dedupe)
                print(f"Importert / Imported {added} -> {ns.file} (hoppet over {skipped} duplikater / skipped duplicates)")
        else:
            print(dataset_stats(load_records(ns.file)))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"FEIL / ERROR: {e}")
        raise SystemExit(1)
