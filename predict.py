"""Bruk den trente hjernen / Inference with the trained brain (100% offline).

Eksempler / Examples:
  python predict.py                                  # demo
  python predict.py "twelve bar blues in Delta style"
  python predict.py --file texts.txt                 # batch: one text per line
  python predict.py --interactive
"""

import argparse
import sys
from typing import List, Union

import torch

from data import Vocab
from embeddings import encoder_from_spec
from model import build_model
from utils import say


class Brain:
    """Last en lagret modell og gjør prediksjoner. / Load a saved model and predict.

    Python-bruk / Python usage:
        from predict import Brain
        brain = Brain("models/brain.pth")
        brain.predict("text")            -> {"label": ..., "confidence": ..., "probabilities": {...}}
        brain.predict_batch(["a", "b"])  -> list of the above
    """

    def __init__(self, path: str = "models/brain.pth"):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fant ikke modell / Model not found: {path}. Kjør/Run: python train.py")
        self.config, self.classes, self.max_len = ckpt["config"], ckpt["classes"], ckpt["max_len"]
        self.vocab = Vocab(ckpt["vocab"]) if ckpt.get("vocab") else None
        self.mode = ckpt.get("mode", "simple")
        # hashed/pretrained: samme encoder som under trening kreves / the training-time encoder is required
        self.encoder = encoder_from_spec(ckpt["encoder"]) if ckpt.get("encoder") else None
        self.checkpoint = ckpt
        self.task = self.config["task"]
        self.model = build_model(self.config)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()  # viktig: slår av dropout/batchnorm-trening / disables dropout, BN uses running stats

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[dict]:
        if not texts:
            return []
        if self.encoder is not None:
            x = self.encoder.encode(list(texts))
        else:
            x = torch.tensor([self.vocab.encode(t, self.max_len) for t in texts], dtype=torch.long)
        out = self.model(x)
        results = []
        if self.task == "regression":
            return [{"value": float(v)} for v in out.squeeze(-1)]
        probs = torch.softmax(out, dim=1)
        for p in probs:
            conf, idx = p.max(0)
            results.append({"label": self.classes[idx], "confidence": float(conf),
                            "probabilities": {c: float(v) for c, v in zip(self.classes, p)}})
        return results

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]


def show(text: str, r: dict) -> None:
    short = text if len(text) <= 70 else text[:67] + "..."
    if "value" in r:
        print(f"  {r['value']:.4f}  <- {short}")
    else:
        print(f"  {r['label']:<22} {r['confidence'] * 100:5.1f}%  <- {short}")


DEMO = [
    "Robert Johnson played slide guitar at the Mississippi crossroads",
    "How to fix a null pointer exception in my Python script",
    "Thanksgiving dinner and Friday night football in a small Texas town",
    "Ways to monetize your blog with affiliate links and ads",
    "A morning routine with meditation and sleep tracking boosts focus",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Bruk hjernen / Use the brain")
    p.add_argument("text", nargs="*", help="Tekst å klassifisere / text(s) to classify")
    p.add_argument("--model", default="models/brain.pth")
    p.add_argument("--file", help="Tekstfil, én tekst per linje / text file, one text per line")
    p.add_argument("--interactive", action="store_true", help="Skriv inn tekst løpende / prompt loop")
    a = p.parse_args(argv)

    try:
        brain = Brain(a.model)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        say(f"FEIL: kunne ikke laste modell: {e}", f"ERROR: could not load model: {e}")
        return 1
    say(f"Modell lastet ({a.model}), oppgave: {brain.task}", f"Model loaded ({a.model}), task: {brain.task}")

    texts: List[str] = list(a.text)
    if a.file:
        try:
            with open(a.file, encoding="utf-8") as f:
                texts += [l.strip() for l in f if l.strip()]
        except OSError as e:
            say(f"FEIL: kunne ikke lese fil: {e}", f"ERROR: could not read file: {e}")
            return 1

    if a.interactive:
        say("Skriv tekst (tom linje avslutter).", "Type text (empty line quits).")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            show(line, brain.predict(line))
        return 0

    if not texts:
        say("Ingen tekst gitt - kjører demo.", "No text given - running demo.")
        texts = DEMO
    for t, r in zip(texts, brain.predict_batch(texts)):
        show(t, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
