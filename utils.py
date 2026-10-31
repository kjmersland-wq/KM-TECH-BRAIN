"""Små hjelpefunksjoner / Small helpers shared by all scripts."""

import random
import sys

import numpy as np
import torch

# Windows-konsoller bruker ofte ikke UTF-8 / Windows consoles often are not UTF-8
for _s in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def say(no: str, en: str) -> None:
    """Skriv melding på norsk og engelsk. / Print a message in Norwegian and English."""
    print(f"{no}\n    [EN] {en}")


def set_seed(seed: int) -> None:
    """Gjør kjøringer reproduserbare. / Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Bruk GPU hvis den finnes, ellers CPU. / Use GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
