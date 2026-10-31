"""Tekst -> vektor / Text-to-vector encoders. 100% lokalt / 100% local.

Tre moduser / Three modes (train.py --mode ...):
  simple      Egen embedding-lag trent fra scratch (standard, ingen ekstra pakker).
              Own embedding layer trained from scratch (default, no extra packages).
  hashed      Lettvekts-alternativ: hashede ord + tegn-n-gram (TF-IDF-lignende), kun numpy/torch.
              Lightweight alternative: hashed words + char n-grams. Handles Norwegian word forms well.
  pretrained  Ferdigtrente setnings-embeddings (sentence-transformers). Best forståelse, støtter norsk
              med standardmodellen. Krever `pip install sentence-transformers`. Faller tilbake til
              `simple` hvis pakken mangler.
              Pre-trained sentence embeddings; falls back to `simple` if the package is missing.

Offline: sentence-transformers laster ned modellen KUN første gang og KUN med --download.
Etter det kjører alt uten nett (modellen ligger i HuggingFace-cachen).
The model is downloaded only once and only with --download; afterwards everything runs offline.
"""

import importlib.util
import os
import re
import zlib
from typing import List, Optional, Tuple

import torch

from utils import say

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # ~120 MB, 50+ språk inkl. norsk / incl. Norwegian
_WORD = re.compile(r"[\w']+", re.UNICODE)


class HashedEncoder:
    """Ingen avhengigheter, ingen nedlasting. Words + char 3-5-grams hashed into `dim` buckets."""

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def spec(self) -> dict:
        return {"type": "hashed", "dim": self.dim}

    def _vec(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim)
        for w in _WORD.findall(text.lower()):
            feats = [("w", w)] + [("c", f"<{w}>"[i:i + n]) for n in (3, 4, 5) for i in range(len(w) + 3 - n)]
            for kind, f in feats:
                h = zlib.crc32(f"{kind}:{f}".encode("utf-8"))
                v[h % self.dim] += (1.0 if (h >> 16) & 1 else -1.0) * (2.0 if kind == "w" else 1.0)
        return v

    def encode(self, texts: List[str]) -> torch.Tensor:
        x = torch.stack([self._vec(t) for t in texts])
        x = torch.sign(x) * torch.log1p(x.abs())  # sublinear tf
        return torch.nn.functional.normalize(x, dim=1)


class SentenceEncoder:
    """Ferdigtrent setnings-embedding via sentence-transformers. / Pre-trained sentence embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL, allow_download: bool = False):
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"  # strengt offline / strictly offline
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            hint = "" if allow_download else (
                f"\nModellen er ikke lastet ned. Kjør én gang med nett: python train.py --mode pretrained --download"
                f"\nModel not cached. Run once with internet: python train.py --mode pretrained --download")
            raise RuntimeError(f"Kunne ikke laste '{model_name}' / could not load: {e}{hint}") from e
        self.dim = self.model.get_sentence_embedding_dimension()

    def spec(self) -> dict:
        return {"type": "sentence", "model_name": self.model_name, "dim": self.dim}

    def encode(self, texts: List[str]) -> torch.Tensor:
        emb = self.model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        return emb.detach().cpu().float()


def sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def encoder_from_spec(spec: dict, allow_download: bool = False):
    """Gjenskap encoder fra checkpoint. / Rebuild the encoder stored in a checkpoint."""
    if spec["type"] == "hashed":
        return HashedEncoder(spec["dim"])
    if spec["type"] == "sentence":
        if not sentence_transformers_available():
            raise RuntimeError("Modellen krever sentence-transformers / this model needs it: "
                               "pip install sentence-transformers")
        return SentenceEncoder(spec["model_name"], allow_download)
    raise ValueError(f"Ukjent encoder / unknown encoder: {spec}")


def load_encoder(mode: str, model_name: str = DEFAULT_MODEL, allow_download: bool = False) -> Tuple[Optional[object], str]:
    """Returnerer (encoder|None, effektiv_modus). `simple` -> (None, 'simple').
    pretrained faller tilbake til simple hvis pakke/modell mangler. / pretrained falls back to simple."""
    if mode == "simple":
        return None, "simple"
    if mode == "hashed":
        return HashedEncoder(), "hashed"
    if mode == "pretrained":
        if not sentence_transformers_available():
            say("Merk: sentence-transformers er ikke installert - bruker enkel embedding (simple). "
                "Installer med: pip install sentence-transformers",
                "Note: sentence-transformers not installed - falling back to the simple embedding. "
                "Install: pip install sentence-transformers")
            return None, "simple"
        try:
            return SentenceEncoder(model_name, allow_download), "pretrained"
        except RuntimeError as e:
            say(f"Merk: {e}\nBruker enkel embedding i stedet.", f"Note: {e}\nFalling back to the simple embedding.")
            return None, "simple"
    raise ValueError(f"Ukjent modus / unknown mode: {mode}")
