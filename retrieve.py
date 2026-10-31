"""Finn lignende tekster / Find similar texts in your own library (embeddings + cosine). 100% local.

  python retrieve.py "slide guitar in open G"                  # topp 5 fra data/music_genres.csv
  python retrieve.py "Miles Davis modal jazz" -k 3
  python retrieve.py "riff i drop D" --data data/example_data.csv --mode hashed
  python retrieve.py --file spørsmål.txt                        # én tekst per linje / one text per line

Moduser / Modes:  auto (pretrained hvis installert og lastet ned, ellers hashed) | hashed | pretrained
auto = pretrained if installed and already downloaded, otherwise hashed.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from data import load_records
from embeddings import DEFAULT_MODEL, HashedEncoder, SentenceEncoder, sentence_transformers_available
from utils import say


class Retriever:
    """Bygger en vektorindeks over eksemplene dine og finner de nærmeste.
    Builds a vector index over your examples and returns the nearest ones.

        r = Retriever("data/music_genres.csv")
        r.similar("slide guitar in open G", k=3)  -> [{"text":..., "label":..., "score":0.61}, ...]
        r.vote("slide guitar in open G")          -> ("Blues", 0.8)   # weighted kNN vote
    """

    def __init__(self, data: str = "data/music_genres.csv", mode: str = "auto",
                 model_name: str = DEFAULT_MODEL, cache_dir: str = "models", quiet: bool = False):
        self.records: List[Tuple[str, str]] = load_records(data)
        self.data = data
        self.encoder, self.mode = self._make_encoder(mode, model_name, quiet)
        self.vectors = self._index(Path(cache_dir))

    @staticmethod
    def _make_encoder(mode: str, model_name: str, quiet: bool):
        if mode in ("auto", "pretrained") and sentence_transformers_available():
            try:
                return SentenceEncoder(model_name, allow_download=False), "pretrained"
            except RuntimeError as e:
                if mode == "pretrained" and not quiet:
                    say(f"Merk: {e}\nBruker hashed i stedet.", f"Note: {e}\nUsing hashed instead.")
        elif mode == "pretrained" and not quiet:
            say("Merk: sentence-transformers er ikke installert, bruker hashed.",
                "Note: sentence-transformers not installed, using hashed.")
        return HashedEncoder(), "hashed"

    def _index(self, cache_dir: Path) -> torch.Tensor:
        p = Path(self.data)
        stamp = f"{p.stat().st_size}:{int(p.stat().st_mtime)}:{self.mode}:{getattr(self.encoder, 'model_name', '')}"
        cache = cache_dir / f"index_{p.stem}_{self.mode}.pt"
        try:
            if cache.exists():
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                if saved["stamp"] == stamp:
                    return saved["vectors"]
        except Exception:
            pass
        vectors = self.encoder.encode([t for t, _ in self.records])
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"stamp": stamp, "vectors": vectors}, cache)
        except OSError:
            pass  # cache er valgfri / the cache is optional
        return vectors

    def similar(self, text: str, k: int = 5, label: Optional[str] = None, min_score: float = 0.0,
                exclude_exact: bool = True) -> List[dict]:
        q = self.encoder.encode([text])[0]
        scores = self.vectors @ q  # normaliserte vektorer -> cosinus / normalized -> cosine
        order = torch.argsort(scores, descending=True).tolist()
        out = []
        for i in order:
            t, l = self.records[i]
            if (label and l != label) or float(scores[i]) < min_score:
                continue
            if exclude_exact and t.strip().lower() == text.strip().lower():
                continue
            out.append({"text": t, "label": l, "score": float(scores[i])})
            if len(out) >= k:
                break
        return out

    def vote(self, text: str, k: int = 5) -> Tuple[Optional[str], float]:
        """Vektet kNN-avstemning: (label, andel av total score). / Weighted kNN vote."""
        hits = [h for h in self.similar(text, k) if h["score"] > 0]
        if not hits:
            return None, 0.0
        tally: dict = {}
        for h in hits:
            tally[h["label"]] = tally.get(h["label"], 0.0) + h["score"]
        best = max(tally, key=tally.get)
        return best, tally[best] / sum(tally.values())

    def evaluate(self, records) -> float:
        """Treffprosent for kNN-avstemning på ferdig merkede tekster. / kNN-vote accuracy on labeled texts."""
        ok = sum(self.vote(t)[0] == l for t, l in records)
        return ok / len(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Finn lignende eksempler / Find similar examples")
    ap.add_argument("text", nargs="*")
    ap.add_argument("--file", help="Tekstfil, én tekst per linje / text file, one per line")
    ap.add_argument("--data", default="data/music_genres.csv")
    ap.add_argument("--mode", choices=["auto", "hashed", "pretrained"], default="auto")
    ap.add_argument("--encoder-model", default=DEFAULT_MODEL)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--label", help="Bare eksempler med denne labelen / only examples with this label")
    ap.add_argument("--eval", help="Test kNN-treff mot en merket fil / test kNN accuracy on a labeled file")
    a = ap.parse_args(argv)

    try:
        r = Retriever(a.data, a.mode, a.encoder_model)
        say(f"Indeks klar: {len(r.records)} eksempler, modus {r.mode}", f"Index ready: {len(r.records)} examples, mode {r.mode}")
        if a.eval:
            recs = load_records(a.eval)
            say(f"kNN-treff på {a.eval}: {r.evaluate(recs):.1%} ({len(recs)} tekster)",
                f"kNN accuracy on {a.eval}: {r.evaluate(recs):.1%} ({len(recs)} texts)")
            return 0
        texts = list(a.text)
        if a.file:
            texts += [l.strip() for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        say(f"FEIL: {e}", f"ERROR: {e}")
        return 1
    if not texts:
        ap.print_help()
        return 0
    for t in texts:
        label, share = r.vote(t, a.k)
        print(f"\n> {t}\n  Mest lik / closest label: {label} ({share:.0%})" if label else f"\n> {t}\n  Ingen treff / no match")
        for h in r.similar(t, a.k, a.label):
            print(f"  {h['score']:.2f}  [{h['label']}] {h['text'][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
