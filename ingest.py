"""Slipp filer i inbox/ og importer dem. / Drop files into inbox/ and import them.

  python ingest.py --init                 # lager mappene / creates the folders
  python ingest.py                        # importerer alt i inbox/ / imports everything in inbox/
  python ingest.py --dry-run              # vis hva som ville skjedd / preview only
  python data.py inbox                    # samme som over / same thing

Struktur / Layout (mappenavn = kategori / folder name = category):
  inbox/music/Blues/notater.txt           -> data/music_genres.csv   [Blues]
  inbox/music/Jazz/anmeldelse.md
  inbox/web/SEO/tips.txt                  -> data/web_topics.csv     [SEO]
  inbox/music/blandet.txt                 -> linjer som  Jazz<TAB>tekst  eller  Jazz | tekst

Hver fil deles automatisk / each file is split automatically:
  * navn.lines.txt          én tekst per linje / one example per line
  * blanke linjer i filen   ett eksempel per avsnitt / one example per paragraph (good for lyric stanzas)
  * kort fil (<=120 ord)    hele filen er ett eksempel / the whole file is one example
  * lang fil                biter på ca. 60 ord ved setningsgrenser / ~60-word chunks at sentence breaks
Ferdige filer flyttes til inbox/_imported/ (bruk --keep for å la dem ligge) / done files move to _imported/.
"""

import argparse
import csv
import difflib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data import TEXT_EXT, _norm, _split_text, load_records
from utils import say

LIBRARIES = {"music": "data/music_genres.csv", "web": "data/web_topics.csv"}
MIN_CHARS = 12  # kortere enn dette er sannsynligvis støy / shorter is probably noise


def split_auto(content: str, filename: str) -> List[str]:
    """Velg fornuftig oppdeling for en tekstfil. / Pick a sensible split for a text file."""
    if filename.lower().endswith(".lines.txt"):
        return _split_text(content, "lines")
    paragraphs = _split_text(content, "paragraphs")
    if len(paragraphs) > 1 and all(len(p.split()) <= 150 for p in paragraphs):
        return paragraphs
    if len(content.split()) <= 120:
        return [content.strip()]
    return _split_text(content, "chunks", 60)


def resolve_label(raw: str, known: Dict[str, str], allow_new: bool) -> Tuple[Optional[str], Optional[str]]:
    """Mappenavn -> label. Rette små skrivefeil / match case. Returnerer (label, advarsel)."""
    raw = raw.replace("__", "/").strip()
    if raw.lower() in known:
        return known[raw.lower()], None
    close = difflib.get_close_matches(raw.lower(), list(known), n=1, cutoff=0.7)
    if close and not allow_new:
        return None, f"'{raw}' er ukjent, mente du/did you mean '{known[close[0]]}'?"
    if known and not allow_new:
        return None, f"'{raw}' er ikke en kjent kategori/not a known category ({', '.join(sorted(known.values()))}). Bruk --allow-new for en ny/for a new one."
    return raw, None


def collect(inbox: Path, domain: str, known: Dict[str, str], allow_new: bool):
    """Gå gjennom inbox/<domain>. Returnerer (eksempler, filer, advarsler). / Walk one domain folder."""
    base = inbox / domain
    examples: List[Tuple[str, str, Path]] = []  # (tekst, label, fil)
    warnings: List[str] = []
    bad_files: set = set()
    for f in sorted(base.rglob("*")):
        if not f.is_file() or "_imported" in f.parts or f.suffix.lower() not in TEXT_EXT | {".csv", ".json"}:
            continue
        rel = f.relative_to(base)
        try:
            if f.suffix.lower() in (".csv", ".json"):
                rows = load_records(str(f))
                for t, l in rows:
                    lab, warn = resolve_label(l, known, allow_new)
                    (examples.append((t, lab, f)) if lab else warnings.append(f"{rel}: {warn}"))
                continue
            content = f.read_text(encoding="utf-8-sig", errors="replace")
            if len(rel.parts) > 1:  # undermappe = label
                lab, warn = resolve_label(rel.parts[0], known, allow_new)
                if not lab:
                    warnings.append(f"{rel}: {warn}")
                    bad_files.add(f)
                    continue
                examples += [(t, lab, f) for t in split_auto(content, f.name)]
            else:  # 'Label<TAB>tekst' eller 'Label | tekst'
                for line in content.splitlines():
                    if "\t" in line or " | " in line:
                        raw, text = line.split("\t", 1) if "\t" in line else line.split(" | ", 1)
                        lab, warn = resolve_label(raw, known, allow_new)
                        (examples.append((text.strip(), lab, f)) if lab else warnings.append(f"{rel}: {warn}"))
                    elif line.strip():
                        warnings.append(f"{rel}: linje uten kategori hoppet over / line without category skipped: {line[:50]}")
        except (ValueError, KeyError, OSError) as e:
            warnings.append(f"{rel}: {e}")
            bad_files.add(f)
    return examples, bad_files, warnings


def append_rows(path: Path, rows: List[Tuple[str, str]]) -> None:
    """Legg til alle rader i én operasjon. / Append all rows in one go."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        data += [{"text": t, "label": l} for t, l in rows]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        if new:
            w.writerow(["text", "label"])
        w.writerows([[t, l] for t, l in rows])


def import_into(path: str, items: List[Tuple[str, str]], dry_run: bool = False, dedupe: bool = True) -> dict:
    """Importer (tekst, label)-par med duplikatsjekk. Returnerer en rapport med tall per kategori.
    Import (text, label) pairs with duplicate check; returns per-category numbers."""
    p = Path(path)
    seen = {_norm(t) for t, _ in load_records(str(p))} if dedupe and p.exists() else set()
    added, dup, short = Counter(), Counter(), Counter()
    rows = []
    for text, lab in items:
        text = " ".join(text.split())
        if len(text) < MIN_CHARS:
            short[lab] += 1
        elif _norm(text) in seen:
            dup[lab] += 1
        else:
            seen.add(_norm(text))
            rows.append((text, lab))
            added[lab] += 1
    if rows and not dry_run:
        append_rows(p, rows)
    total = len(load_records(str(p))) if p.exists() else 0
    return {"path": str(p), "added": added, "dup": dup, "short": short, "total": total + (len(rows) if dry_run else 0)}


def print_report(name: str, rep: dict, dry_run: bool = False) -> None:
    a, d, s = sum(rep["added"].values()), sum(rep["dup"].values()), sum(rep["short"].values())
    tag = " (prøvekjøring/dry run)" if dry_run else ""
    print(f"\n== {name} -> {rep['path']}{tag}")
    print(f"   {a} lagt til/added, {d} duplikater/duplicates, {s} for korte/too short. Totalt nå/total now: {rep['total']}")
    for lab in sorted(set(rep["added"]) | set(rep["dup"]) | set(rep["short"])):
        extra = ", ".join(x for x in (f"{rep['dup'][lab]} dup" if rep["dup"][lab] else "",
                                       f"{rep['short'][lab]} kort/short" if rep["short"][lab] else "") if x)
        print(f"   {lab:<24} +{rep['added'][lab]:<4}{'(' + extra + ')' if extra else ''}")
    if rep["path"] and Path(rep["path"]).exists():
        counts = Counter(l for _, l in load_records(rep["path"]))
        low = [f"{l} ({n})" for l, n in counts.items() if n < 20]
        if low:
            print(f"   Tips: under 20 eksempler / fewer than 20: {', '.join(low)}")


def init_folders(inbox: Path) -> None:
    for domain, path in LIBRARIES.items():
        labels = sorted({l for _, l in load_records(path)}) if Path(path).exists() else []
        for lab in labels:
            (inbox / domain / lab.replace("/", "__")).mkdir(parents=True, exist_ok=True)
    say(f"Mappene er klare i {inbox}/. Legg .txt/.md-filer i riktig undermappe og kjør: python ingest.py",
        f"Folders ready in {inbox}/. Put .txt/.md files in the right subfolder and run: python ingest.py")
    for domain in LIBRARIES:
        d = inbox / domain
        if d.exists():
            print(f"   {d.as_posix()}/: " + ", ".join(sorted(p.name for p in d.iterdir() if p.is_dir())))


def run(inbox: str = "inbox", dry_run: bool = False, keep: bool = False, allow_new: bool = False) -> int:
    root = Path(inbox)
    if not root.exists():
        say(f"Fant ikke {root}/. Kjør først: python ingest.py --init", f"No {root}/ yet. Run first: python ingest.py --init")
        return 1
    stray = [p.name for p in root.iterdir() if p.is_file() or (p.is_dir() and p.name not in LIBRARIES and not p.name.startswith("_"))]
    if stray:
        say(f"Hopper over ting utenfor music/ og web/: {', '.join(stray)}", f"Skipping things outside music/ and web/: {', '.join(stray)}")
    any_work, all_warnings, moved = False, [], 0
    for domain, lib in LIBRARIES.items():
        if not (root / domain).exists():
            continue
        known = {l.lower(): l for _, l in load_records(lib)} if Path(lib).exists() else {}
        items, bad, warns = collect(root, domain, known, allow_new)
        all_warnings += [f"[{domain}] {w}" for w in warns]
        if not items:
            continue
        any_work = True
        rep = import_into(lib, [(t, l) for t, l, _ in items], dry_run)
        print_report(domain, rep, dry_run)
        if not dry_run and not keep:
            done = {f for _, _, f in items} - bad
            for f in done:
                dest = root / "_imported" / f.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest.with_name(dest.name) if not dest.exists() else dest.with_name(f"{dest.stem}_{moved}{dest.suffix}")))
                moved += 1
    for w in all_warnings:
        print(f"  ! {w}")
    if not any_work:
        say("Ingenting å importere. Legg filer i inbox/music/<Sjanger>/ eller inbox/web/<Tema>/.",
            "Nothing to import. Put files in inbox/music/<Genre>/ or inbox/web/<Topic>/.")
    elif not dry_run:
        say(f"Ferdig. {moved} filer flyttet til {root}/_imported/. Søk og agenter bruker de nye eksemplene med en gang; "
            f"vil du trene sjangermodellen på nytt: python ingest.py --retrain",
            f"Done. {moved} files moved to {root}/_imported/. Search and agents use the new examples right away; "
            f"to retrain the genre model: python ingest.py --retrain")
    return 0


def retrain_music() -> int:
    import train
    return train.main(["--data", LIBRARIES["music"], "--mode", "hashed", "--out", "models/genres.pth",
                       "--epochs", "400", "--patience", "60", "--log-every", "1000"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Importer fra inbox/ / Import from inbox/")
    ap.add_argument("--inbox", default="inbox")
    ap.add_argument("--init", action="store_true", help="Lag mappene / create the folders")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true", help="Ikke flytt ferdige filer / do not move imported files")
    ap.add_argument("--allow-new", action="store_true", help="Tillat nye kategorier / allow new categories")
    ap.add_argument("--retrain", action="store_true", help="Tren sjangermodellen på nytt etterpå / retrain the genre model afterwards")
    a = ap.parse_args(argv)
    try:
        if a.init:
            init_folders(Path(a.inbox))
            return 0
        code = run(a.inbox, a.dry_run, a.keep, a.allow_new)
        if code == 0 and a.retrain and not a.dry_run:
            return retrain_music()
        return code
    except (FileNotFoundError, ValueError, KeyError) as e:
        say(f"FEIL: {e}", f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
