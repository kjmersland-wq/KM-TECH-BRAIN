"""Agent-lag / Agent layer on top of the trained brain. 100% local, no APIs.

En agent = et lite fagfelt med nøkkelord, verktøy ("tasks") og valgfri bruk av den trente modellen.
An agent = a small specialist with keywords, tools ("tasks") and optional use of the trained model.
The agents are rule/template based - they do NOT generate free text. The model classifies, the agent acts.

Bruk / Usage:
    python agents.py --list
    python agents.py "ideas for a blues blog about Delta slide guitar"        # auto-routes to an agent
    python agents.py --agent web "SEO checklist for my Next.js shop"
    python agents.py --interactive

Legg til en ny agent / Add a new agent: subclass Agent, fill name/description/keywords/domains,
add methods to `tasks`, then call register(MyAgent). That's it.
"""

import argparse
import re
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional

from data import tokenize
from utils import say

STOP = set("""a an the and or of to in on for with is are was be it this that my your our how what why when
about from by at as i you we they me us can do does into over under up out not no så og i på for med er det en
et å av til som jeg du vi de min din vår hvordan hva hvorfor om fra ikke have has had will would should could
make get use using idea ideas checklist content analyze analyse tips tip set setup should need want
please give show write help best good new""".split())


def keywords_of(text: str, n: int = 6) -> List[str]:
    """Viktigste ord i en tekst. / Most frequent meaningful words in a text."""
    words = [w for w in tokenize(text) if w not in STOP and len(w) > 2 and not w.isdigit()]
    return [w for w, _ in Counter(words).most_common(n)]


def topic_of(text: str, fallback: str = "your topic") -> str:
    k = keywords_of(text, 3)
    return " ".join(k) if k else fallback


# ------------------------------------------------------------------ base class

class Agent:
    name = "base"
    description = ""
    keywords: Dict[str, float] = {}      # keyword -> weight (substring match, lowercase)
    domains: Dict[str, float] = {}       # model label -> weight (how much this label hints at this agent)
    task_triggers: Dict[str, List[str]] = {}  # task name -> trigger words

    def __init__(self, brain=None):
        self.brain = brain  # predict.Brain or None (agents work without a model, just less smart)

    # --- routing
    def relevance(self, text: str, cls: Optional[dict] = None) -> float:
        low = text.lower()
        score = sum(w for k, w in self.keywords.items() if k in low)
        if cls and "label" in cls:
            score += self.domains.get(cls["label"], 0.0) * cls["confidence"] * 2
        return score

    def pick_tasks(self, text: str, limit: int = 3) -> List[str]:
        """Alle oppgaver med treff, beste først. / All tasks with trigger hits, best first."""
        low = text.lower()
        hits = {t: sum(w in low for w in ws) for t, ws in self.task_triggers.items()}
        ranked = [t for t, n in sorted(hits.items(), key=lambda kv: -kv[1]) if n > 0]
        return ranked[:limit] or ["advice"]

    # --- running
    def classify(self, text: str) -> Optional[dict]:
        return self.brain.predict(text) if self.brain else None

    def run(self, text: str, task: Optional[str] = None) -> dict:
        sections = []
        for t in ([task] if task else self.pick_tasks(text)):
            fn: Callable[[str], List[str]] = getattr(self, f"task_{t}", None)
            if fn is None:
                raise ValueError(f"{self.name} har ikke oppgaven / has no task '{t}'")
            sections.append((t, fn(text)))
        return {"agent": self.name, "classification": self.classify(text), "sections": sections}

    def task_advice(self, text: str) -> List[str]:
        return [f"I can help with: {', '.join(t for t in self.task_triggers)}.",
                f"Try e.g. '{self.name}: {next(iter(self.task_triggers))} for {topic_of(text)}'."]


# ------------------------------------------------------------------ MusicAgent

class MusicAgent(Agent):
    name = "music"
    description = "Blues, modern music, lyrics analysis, metadata and content ideas."
    keywords = {"blues": 3, "guitar": 2, "song": 2, "lyric": 3, "music": 2, "album": 2, "playlist": 2, "chord": 2,
                "slide": 1, "harmonica": 2, "band": 1, "gitar": 2, "låt": 2, "tekst": 1, "musikk": 2,
                "delta": 1, "jazz": 1, "track": 1, "artist": 1, "spotify": 2, "riff": 2, "verse": 1, "chorus": 1}
    domains = {"Blues/Roots": 3.0, "American Culture": 0.5}
    task_triggers = {"lyrics": ["lyric", "verse", "stanza", "tekst", "analyze", "analyse"],
                     "metadata": ["metadata", "tags", "title", "description", "spotify", "release", "tittel"],
                     "ideas": ["idea", "content", "blog", "video", "post", "idé", "innhold"]}

    def task_ideas(self, text: str) -> List[str]:
        t = topic_of(text, "the blues")
        return [f"Blog: '5 things every beginner should know about {t}'",
                f"Blog: 'The story behind {t}' (history + one listening list)",
                f"Short video: 60-second riff or lick lesson on {t}",
                f"Playlist: '{t.title()} - essentials and deep cuts' with 2 lines of liner notes per track",
                f"Series: 'Roots to modern' - trace {t} from Delta/Chicago to today's artists",
                f"Newsletter: monthly 'Track of the month' featuring {t}",
                "Tip: pair each idea with one clear call to action (listen, subscribe, download tab)."]

    def task_lyrics(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 3:  # allow ';' or '/' separated one-liners
            lines = [l.strip() for l in re.split(r"[;/|]", text) if l.strip()]
        if len(lines) < 2:
            return ["Paste lyrics with one line per row (or separate lines with ';') to analyze them."]
        norm = [re.sub(r"[^\w\s]", "", l.lower()) for l in lines]
        rep = len(norm) - len(set(norm))
        words = [w for w in tokenize(" ".join(lines), ngrams=False) if w not in STOP]
        ends = [n.split()[-1][-2:] for n in norm if n.split()]
        rhymes = sum(c - 1 for c in Counter(ends).values() if c > 1)
        aab = sum(1 for i in range(0, len(norm) - 2, 3) if norm[i] == norm[i + 1] != norm[i + 2])
        out = [f"{len(lines)} lines, {len(words)} content words, {len(set(words))} unique.",
               f"Repeated lines: {rep} ({rep / len(lines):.0%}) - repetition is typical for blues and hooks.",
               f"Line-ending rhyme hits (last 2 letters): {rhymes}.",
               f"Top words: {', '.join(keywords_of(' '.join(lines), 5))}."]
        out.append("AAB blues stanza structure detected." if aab else
                   "No AAB stanza (line 1 = line 2, line 3 answers) found - try it for a classic blues feel.")
        return out

    def task_metadata(self, text: str) -> List[str]:
        k = keywords_of(text, 5)
        cls = self.classify(text)
        genre = cls["label"] if cls else "Music"
        first = re.split(r"(?<=[.!?])\s", text.strip())[0] if text.strip() else ""
        return [f"Title ideas: '{' '.join(k[:3]).title()}' | '{(k[0] if k else 'Untitled').title()} Blues' | "
                f"'Notes on {topic_of(text).title()}'",
                f"Tags: {', '.join(k + [genre.lower().replace('/', ' ')])}",
                f"Description (<=155 chars): {first[:155]}",
                "Checklist: artist, title, genre, key/BPM, year, ISRC, cover art 3000x3000, lyrics credit."]


# ------------------------------------------------------------------ WebAgent

class WebAgent(Agent):
    name = "web"
    description = "Web projects, SEO, Next.js, Stripe and technical decisions."
    keywords = {"seo": 3, "next.js": 3, "nextjs": 3, "react": 2, "stripe": 3, "website": 2, "web": 1, "html": 1,
                "css": 1, "deploy": 2, "cloudflare": 3, "vercel": 2, "api": 1, "checkout": 2, "payment": 2,
                "landing": 2, "domain": 1, "nettside": 2, "javascript": 1, "typescript": 1, "database": 1,
                "hosting": 2, "ssr": 2, "static": 1}
    domains = {"Code/Tech": 2.5, "Business/Monetization": 1.5}
    task_triggers = {"seo": ["seo", "search engine", "ranking", "meta", "sitemap", "google"],
                     "stripe": ["stripe", "payment", "checkout", "subscription", "betaling"],
                     "nextjs": ["next.js", "nextjs", "app router", "ssr", "react"],
                     "decision": ["should i", "which", "vs", "choose", "decision", "hosting", "deploy", "cloudflare", "velge"]}

    def task_seo(self, text: str) -> List[str]:
        out = ["Title 50-60 chars with main keyword first; meta description 120-155 chars.",
               "One <h1> per page; logical <h2>/<h3>; descriptive alt text; clean URLs.",
               "sitemap.xml + robots.txt; canonical tags; Open Graph tags; fast LCP (<2.5s) and mobile-friendly.",
               "Internal links between related pages; structured data (JSON-LD) where it fits."]
        words = tokenize(text, ngrams=False)
        if len(words) >= 40:  # looks like page copy -> analyze it
            sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
            avg = len(words) / max(1, len(sents))
            out += [f"Your text: {len(words)} words, avg sentence {avg:.1f} words"
                    f"{' (long - split some)' if avg > 25 else ''}.",
                    f"Likely focus keywords: {', '.join(keywords_of(text, 5))}."]
        return out

    def task_stripe(self, text: str) -> List[str]:
        return ["Use Stripe Checkout (hosted) first; build custom Payment Element only if you must.",
                "Create Checkout Sessions on the server; never put the secret key in browser code.",
                "Fulfil orders from a verified webhook (check the signature), not from the success redirect.",
                "Handle webhooks idempotently (store event ids); return 2xx fast.",
                "Develop in test mode with test cards; switch keys only at launch.",
                "Subscriptions: use Billing + Customer Portal for cancel/upgrade instead of building it yourself."]

    def task_nextjs(self, text: str) -> List[str]:
        return ["Default: App Router + Server Components; add 'use client' only where you need state/effects.",
                "Mostly static content? Use static generation/ISR; reserve SSR for per-request personalized pages.",
                "Cloudflare Pages: pure static export (output: 'export') works as-is; for SSR/API routes use the "
                "Cloudflare adapter and check which Node APIs you rely on.",
                "Put secrets in server-only env vars; only NEXT_PUBLIC_* reaches the browser.",
                "Use next/image + next/font and the Metadata API for SEO basics."]

    def task_decision(self, text: str) -> List[str]:
        low = text.lower()
        out = ["Rule of thumb: pick the simplest thing that meets today's requirement.",
               "Content site / landing page -> plain static HTML or a static generator (cheap, fast, easy on Cloudflare Pages).",
               "Interactive app with logins/data -> Next.js (or similar) + managed database + hosted auth.",
               "Payments -> Stripe Checkout; do not store card data yourself."]
        if "cloudflare" in low:
            out.append("Cloudflare Pages: great for static + edge functions; set build output dir (e.g. 'web').")
        if "vercel" in low:
            out.append("Vercel: best-fit for Next.js SSR/ISR with zero config.")
        return out


# ------------------------------------------------------------------ registry / router

REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    REGISTRY[cls.name] = cls
    return cls


register(MusicAgent)
register(WebAgent)


class Router:
    """Velger beste agent for en tekst. / Picks the best agent for a text (model label + keywords)."""

    def __init__(self, brain=None):
        self.brain = brain
        self.agents = {n: c(brain) for n, c in REGISTRY.items()}

    def route(self, text: str) -> Agent:
        cls = self.brain.predict(text) if self.brain else None
        scores = {n: a.relevance(text, cls) for n, a in self.agents.items()}
        best = max(scores, key=scores.get)
        if scores[best] < 1.0:  # for svakt signal -> ikke gjett / too weak a signal, don't guess
            raise LookupError("Fant ingen passende agent / no matching agent. Use --agent or --list.")
        return self.agents[best]

    def run(self, text: str, agent: Optional[str] = None, task: Optional[str] = None) -> dict:
        if agent:
            if agent not in self.agents:
                raise KeyError(f"Ukjent agent / unknown agent '{agent}'. Available: {', '.join(self.agents)}")
            return self.agents[agent].run(text, task)
        return self.route(text).run(text, task)


def load_brain(path: str):
    try:
        from predict import Brain
        return Brain(path)
    except Exception as e:  # missing model / torch -> agents still work with keywords
        say(f"Merk: kjører uten modell ({e}). Kun nøkkelord brukes.",
            f"Note: running without model ({e}). Keywords only.")
        return None


def show(res: dict) -> None:
    c = res["classification"]
    cls = f"  [model: {c['label']} {c['confidence']:.0%}]" if c else ""
    print(f"\n== {res['agent']} agent{cls}")
    for task, lines in res["sections"]:
        print(f"  [{task}]")
        for line in lines:
            print(f"    - {line}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="KM-TECH BRAIN agents (lokalt / local)")
    p.add_argument("text", nargs="*")
    p.add_argument("--agent", choices=sorted(REGISTRY), help="Velg agent / force an agent")
    p.add_argument("--task", help="Velg oppgave / force a task (e.g. ideas, lyrics, metadata, seo, stripe)")
    p.add_argument("--model", default="models/brain.pth")
    p.add_argument("--list", action="store_true")
    p.add_argument("--interactive", action="store_true")
    a = p.parse_args(argv)

    if a.list:
        for n, c in REGISTRY.items():
            print(f"{n:8} {c.description}\n         tasks: {', '.join(c.task_triggers)}")
        return 0

    router = Router(load_brain(a.model))

    def go(text: str) -> None:
        try:
            show(router.run(text, a.agent, a.task))
        except (LookupError, KeyError, ValueError) as e:
            say(f"FEIL: {e}", f"ERROR: {e}")

    if a.interactive:
        say("Skriv en oppgave (tom linje avslutter).", "Type a request (empty line quits).")
        while (line := input("> ").strip()):
            go(line)
        return 0
    if not a.text:
        p.print_help()
        return 0
    go(" ".join(a.text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
