"""Spør hjernen / Ask the brain. Det enkleste startpunktet i hverdagen. / The easiest everyday entry point.

  python ask.py "slide i åpen D på en gammel resonator"     # ett spørsmål / one question
  python ask.py                                              # samtale med minne / conversation with memory

Du får: sjanger eller tema, lignende eksempler fra dine egne data, og en kort tanke.
You get: genre or topic, similar examples from your own data, and a short take.
I samtalen / in the conversation:  mer / more = hele svaret fra agenten, /historikk, /glem, tom linje avslutter.
Alt kjører lokalt og offline. / Everything runs locally and offline.
"""

import argparse
import sys

from agents import brief, build_router, show
from utils import say


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Spør hjernen / Ask the brain")
    p.add_argument("text", nargs="*", help="Spørsmål eller tekst / question or text")
    p.add_argument("--agent", choices=["music", "web", "code"], help="Tving en agent / force an agent (code = lokal LLM via Ollama)")
    p.add_argument("--full", action="store_true", help="Vis hele agentsvaret / show the full agent answer")
    p.add_argument("--model", default="models/brain.pth")
    p.add_argument("--library", default="data/music_genres.csv")
    p.add_argument("--web-library", default="data/web_topics.csv")
    p.add_argument("--genre-model", default="models/genres.pth")
    a = p.parse_args(argv)

    router = build_router(a.model, a.library, a.web_library, a.genre_model)
    last = {"text": None, "agent": None}

    def ask(text: str) -> None:
        try:
            res = router.run(text, a.agent or (last["agent"] if text == last["text"] else None))
        except (LookupError, KeyError, ValueError) as e:
            if "Ollama" in str(e):
                print(f"\n  {e}")
                return
            say(f"Jeg ble ikke klok på den. Prøv med flere ord om musikk eller web. ({e})",
                f"I couldn't place that one. Try a few more words about music or web. ({e})")
            return
        last.update(text=text, agent=res["agent"])
        (show if a.full else brief)(res)

    if a.text:
        ask(" ".join(a.text))
        return 0

    say("Hva lurer du på? (musikk eller web, norsk eller engelsk). «mer» gir hele svaret, tom linje avslutter.",
        "What are you wondering about? (music or web, Norwegian or English). \"more\" gives the full answer, empty line quits.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if line.lower() in ("/historikk", "/history"):
            for m in router.memory.items:
                print(f"  {m['agent']}: {m['text'][:80]}")
        elif line.lower() in ("/glem", "/reset"):
            router.memory.clear()
            print("  (minnet er tømt / memory cleared)")
        elif line.lower() in ("mer", "more") and last["text"]:
            res = router.run(last["text"], last["agent"])  # samme spørsmål, hele svaret / same question, full answer
            show(res)
        else:
            ask(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
