"""Agent-lag / Agent layer on top of the trained brain. 100% local, no APIs.

En agent = et lite fagfelt med nøkkelord, oppgaver ("tasks"), forslag per kategori og valgfri bruk av
den trente modellen. Modellen klassifiserer, agenten handler. Teksten agentene gir er håndskrevet
(ikke generert): personlig, varm og konkret, på norsk eller engelsk etter hva du skriver til dem.
An agent = keywords + tasks + suggestions per category + optional use of the trained model.
The model classifies, the agent acts. The wording is hand-written (not generated): personal, warm,
specific, in Norwegian or English depending on what you write.

Bruk / Usage:
    python agents.py --list
    python agents.py "ideer til en bluesblogg om Delta-gitar"          # velger agent selv / auto-routes
    python agents.py --agent web "SEO for nettbutikken min i Next.js"
    python agents.py --interactive                                      # husker de siste 5 meldingene
    python agents.py --selftest                                         # sjekker stilen / style check

Ny agent / New agent: subclass Agent, fyll name/keywords/domains/task_triggers/suggestions, legg til
`task_<navn>(self, text, c)`-metoder (c = kontekst) og kall register(MinAgent).
"""

import argparse
import re
import sys
import zlib
from collections import Counter, deque
from typing import Dict, List, Optional

from data import tokenize
from utils import say

Text = Dict[str, str]  # {"en": ..., "no": ...}


def L(en: str, no: str) -> Text:
    return {"en": en, "no": no}


# ------------------------------------------------------------------ språk / language

_NO = set("""og ikke jeg det er som på til for med en et av om fra har vi du min mitt mine hvordan hva hvorfor
skal kan vil må noen mye litt også når hvor hvilken hvilke eller men så der her denne dette bli blir ble
sitter gjør gjøre fått får ha være nå bare meg deg oss dem seg""".split())
_EN = set("""the and is to of in for with how my you are that this on it be can do does what why when where which
or but so there here have has not just your our about from into i we they""".split())


def lang_of(text: str) -> str:
    """Norsk eller engelsk, enkel telling. / Norwegian or English, simple word counting."""
    words = tokenize(text, ngrams=False)
    no = sum(w in _NO for w in words) + 2 * len(re.findall(r"[æøåÆØÅ]", text))
    en = sum(w in _EN for w in words)
    return "no" if no > en else "en"


STOP = _NO | _EN | set("""a an i me us idea ideas idé ideer checklist sjekkliste content innhold analyze analyse analyser
tips tip set setup should need want please give show write help best good new more flere mer lag lage skriv skrive
foreslå suggest make get use using make about like noe ting""".split())


def keywords_of(text: str, n: int = 6) -> List[str]:
    words = [w for w in tokenize(text, ngrams=False) if w not in STOP and len(w) > 2 and not w.isdigit()]
    return [w for w, _ in Counter(words).most_common(n)]


def topic_of(text: str) -> str:
    return " ".join(keywords_of(text, 3))


def pick(options: List[Text], seed: str, n: int) -> List[Text]:
    """Deterministisk 'tilfeldig' utvalg så svarene varierer fra tekst til tekst.
    Deterministic pseudo-random pick so answers vary per text but stay reproducible."""
    start = zlib.crc32(seed.encode("utf-8")) % max(1, len(options))
    return [options[(start + i) % len(options)] for i in range(min(n, len(options)))]


# ------------------------------------------------------------------ minne / memory

class Memory:
    """De siste N meldingene i denne økten. / The last N interactions of this session."""

    def __init__(self, size: int = 5):
        self.items: deque = deque(maxlen=size)

    def add(self, **entry) -> None:
        self.items.append(entry)

    def last(self) -> Optional[dict]:
        return self.items[-1] if self.items else None

    def clear(self) -> None:
        self.items.clear()

    def is_followup(self, text: str) -> bool:
        """'mer', 'what about seo?', korte oppfølgere. / short follow-up messages."""
        low = text.lower().strip()
        starts = ("more", "another", "and ", "also", "what about", "how about", "flere", "mer ", "og ",
                  "hva med", "hvordan med", "ok", "okay", "ja", "yes", "men ")
        return bool(self.items) and (len(low.split()) <= 3 or low.startswith(starts))


# ------------------------------------------------------------------ musikk-kunnskap / music knowledge

GENRES = {
    "blues": {
        "kw": ["blues", "delta", "slide", "bottleneck", "twelve bar", "12 bar", "harp", "harmonica", "juke", "chicago blues",
               "piedmont", "resonator", "muddy", "howlin", "robert johnson", "skip james", "b.b. king", "bb king"],
        "listen": [
            L("Skip James, \"Hard Time Killing Floor Blues\" (1931). That open D minor tuning and the falsetto still make the hair on my arms stand up.",
              "Skip James, \"Hard Time Killing Floor Blues\" (1931). Den åpne d-moll-stemmingen og falsetten gir meg fortsatt gåsehud."),
            L("Muddy Waters, \"Hard Again\" (1977). Johnny Winter produced it and basically just let the band be loud and loose. It works.",
              "Muddy Waters, \"Hard Again\" (1977). Johnny Winter produserte, og lot i bunn og grunn bandet få være høyt og løst. Det funker."),
            L("B.B. King, \"Live at the Regal\" (1965). Listen to the crowd. They answer every bent note like it's a conversation.",
              "B.B. King, \"Live at the Regal\" (1965). Hør på publikum. De svarer på hver eneste bend som om det var en samtale."),
            L("Mississippi John Hurt, \"Avalon Blues\". The thumb never stops, and he sounds like he's smiling the whole way through.",
              "Mississippi John Hurt, \"Avalon Blues\". Tommelen stopper aldri, og han høres ut som han smiler hele veien."),
            L("Junior Wells, \"Hoodoo Man Blues\" (1965). A Chicago bar band caught in a room, harp right in your face.",
              "Junior Wells, \"Hoodoo Man Blues\" (1965). Et Chicago-band fanget i ett rom, med munnspillet midt i fjeset ditt."),
        ],
        "hooks": [
            L("Why Skip James still sounds like nobody else", "Hvorfor Skip James fortsatt høres ut som ingen andre"),
            L("The one blues record I'd save from a fire, and what the second bar does to me", "Bluesplata jeg ville reddet fra en brann, og hva takt to gjør med meg"),
            L("Twelve bars, three chords, and a lifetime of getting it wrong: what I've learned playing slide", "Tolv takter, tre akkorder og et helt liv med å ta feil: det jeg har lært av å spille slide"),
        ],
        "closing": L("Best late at night, a notch louder than is polite.", "Best sent på kvelden, en tanke høyere enn naboene liker."),
    },
    "jazz": {
        "kw": ["jazz", "bebop", "coltrane", "miles davis", "monk", "swing", "saxophone", "saksofon", "improvis", "trio", "quartet",
               "kvartett", "bill evans", "ecm", "garbarek", "molde"],
        "listen": [
            L("Miles Davis, \"Kind of Blue\" (1959). Put it on once and don't do anything else. Not even dishes.",
              "Miles Davis, \"Kind of Blue\" (1959). Sett den på én gang og ikke gjør noe annet. Ikke engang oppvasken."),
            L("Bill Evans Trio, \"Waltz for Debbie\" (1961), live at the Village Vanguard. You can hear glasses clinking, and somehow that makes it better.",
              "Bill Evans Trio, \"Waltz for Debbie\" (1961), live fra Village Vanguard. Du hører glass som klirrer, og på en eller annen måte blir det bare bedre."),
            L("John Coltrane, \"A Love Supreme\" (1965). Four parts, one long prayer. Give it forty minutes.",
              "John Coltrane, \"A Love Supreme\" (1965). Fire deler, én lang bønn. Gi den førti minutter."),
            L("Jan Garbarek and the Hilliard Ensemble, \"Officium\" (1994). A saxophone drifting through old vocal music in a church. Very Norwegian, very quiet.",
              "Jan Garbarek og Hilliard Ensemble, \"Officium\" (1994). Saksofon som svever gjennom gammel korsang i en kirke. Veldig norsk, veldig stille."),
        ],
        "hooks": [
            L("The first jazz record that made me stop talking in the middle of a sentence", "Den første jazzplata som fikk meg til å slutte å snakke midt i en setning"),
            L("What a rhythm section actually does (and why you only notice it when it's gone)", "Hva et kompband egentlig gjør (og hvorfor du bare merker det når det mangler)"),
            L("A beginner's route into jazz that doesn't start with 'Take Five'", "En nybegynnerrute inn i jazzen som ikke starter med \"Take Five\""),
        ],
        "closing": L("Wine optional, lights low, phone in another room.", "Vin er valgfritt. Dempet lys og telefonen i et annet rom."),
    },
    "rock": {
        "kw": ["rock", "riff", "distortion", "forvrengning", "stratocaster", "les paul", "hendrix", "zeppelin", "stones", "punk",
               "metal", "garage", "amp", "forsterker", "power chord", "sabbath"],
        "listen": [
            L("Jimi Hendrix, \"Electric Ladyland\" (1968). That's what a Strat can do when nobody tells it no.",
              "Jimi Hendrix, \"Electric Ladyland\" (1968). Sånn låter en Strat når ingen sier nei."),
            L("Led Zeppelin IV (1971). Start with \"When the Levee Breaks\": an old Memphis Minnie blues, turned up until the walls move.",
              "Led Zeppelin IV (1971). Start med \"When the Levee Breaks\", en gammel Memphis Minnie-blues skrudd opp til veggene beveger seg."),
            L("The Rolling Stones, \"Exile on Main St.\" (1972). Recorded in a sweaty basement in France, and you can hear every degree of it.",
              "The Rolling Stones, \"Exile on Main St.\" (1972). Spilt inn i en svett kjeller i Frankrike, og du hører hver eneste grad."),
        ],
        "hooks": [
            L("Three chords and a bad amp: the riff that made me pick up a guitar", "Tre akkorder og en dårlig forsterker: riffet som fikk meg til å plukke opp en gitar"),
            L("Why the best rock guitarists were all secretly blues players", "Hvorfor de beste rockegitaristene egentlig var bluesspillere på hemmelig oppdrag"),
            L("My first garage band, and the song we never learned to play right", "Det første garasjebandet mitt, og låten vi aldri lærte å spille riktig"),
        ],
        "closing": L("Turn it up. Then a bit more.", "Skru opp. Og litt til."),
    },
    "any": {
        "kw": [],
        "listen": [
            L("Pick one record and stay with it for a week before you move on. You hear different things on day five than on day one.",
              "Velg én plate og hold deg til den en hel uke før du går videre. Du hører andre ting på dag fem enn på dag én."),
            L("Ask someone whose taste you trust for the record they'd save from a fire. It beats any algorithm.",
              "Spør noen du stoler på smaken til om hvilken plate de ville reddet fra en brann. Det slår alle algoritmer."),
        ],
        "hooks": [
            L("The record I keep going back to, and why I can't explain it properly", "Plata jeg stadig vender tilbake til, og hvorfor jeg ikke klarer å forklare det ordentlig"),
            L("Five songs that taught me more than any lesson did", "Fem låter som lærte meg mer enn noen musikktime"),
        ],
        "closing": L("Good music is better when it's shared.", "God musikk blir bedre når den deles."),
    },
}

BANNED = ["delve", "unlock", "journey", "tapestry", "game-changer", "game changer", "elevate", "in today's", "dive into",
          "seamless", "leverage", "unleash", "embark", "landscape", "realm", "furthermore", "moreover", "in conclusion",
          "ultimate guide", "whether you're", "i dagens", "utforsk", "spennende reise", "neste nivå", "frigjør",
          "dykk ned", "ta steget", "hos oss finner du"]


def detect_genre(text: str, label: Optional[str] = None, prev: Optional[str] = None) -> str:
    low = text.lower()
    scores = {g: sum(k in low for k in d["kw"]) for g, d in GENRES.items() if g != "any"}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    if prev and prev in GENRES:
        return prev
    return "blues" if label == "Blues/Roots" else "any"


# ------------------------------------------------------------------ base class

class Agent:
    name = "base"
    description = ""
    keywords: Dict[str, float] = {}
    domains: Dict[str, float] = {}
    task_triggers: Dict[str, List[str]] = {}
    suggestions: Dict[str, Text] = {}  # kategori -> neste steg / category -> next step

    def __init__(self, brain=None):
        self.brain = brain

    def keyword_score(self, text: str) -> float:
        low = text.lower()
        return sum(w for k, w in self.keywords.items() if k in low)

    def relevance(self, text: str, cls: Optional[dict] = None) -> float:
        score = self.keyword_score(text)
        if cls and "label" in cls:
            score += self.domains.get(cls["label"], 0.0) * cls["confidence"] * 2
        return score

    def pick_tasks(self, text: str, limit: int = 3) -> List[str]:
        low = text.lower()
        hits = {t: sum(w in low for w in ws) for t, ws in self.task_triggers.items()}
        ranked = [t for t, n in sorted(hits.items(), key=lambda kv: -kv[1]) if n > 0]
        return ranked[:limit]

    def context(self, text: str, cls: Optional[dict], prev: Optional[dict], follow: bool) -> dict:
        topic = topic_of(text) or (prev["topic"] if prev else "")
        if follow and prev and not topic_of(text):
            topic = prev["topic"]
        label = cls["label"] if cls else (prev["label"] if follow and prev else None)
        return {"lang": lang_of(text) if not (follow and len(text.split()) < 3 and prev) else prev["lang"],
                "topic": topic, "label": label, "cls": cls, "genre": detect_genre(text, label, prev.get("genre") if prev else None),
                "seed": text + (prev["text"] if follow and prev else "")}  # oppfølger gir nye forslag / follow-ups vary

    def run(self, text: str, task: Optional[str] = None, prev: Optional[dict] = None, follow: bool = False) -> dict:
        short_follow = follow and prev and len(text.split()) <= 3  # "more" says nothing about the category
        cls = None if (short_follow or not self.brain) else self.brain.predict(text)
        c = self.context(text, cls, prev, follow)
        tasks = [task] if task else (self.pick_tasks(text) or ([prev["tasks"][0]] if follow and prev and prev.get("agent") == self.name else []))
        sections = []
        for t in tasks:
            fn = getattr(self, f"task_{t}", None)
            if fn is None:
                raise ValueError(f"{self.name} har ikke oppgaven / has no task '{t}'")
            sections.append((t, fn(text, c)))
        if not sections:
            sections.append(("advice", self.task_advice(text, c)))
        return {"agent": self.name, "lang": c["lang"], "classification": cls, "sections": sections,
                "next": self.next_step(c), "unsure": self.unsure(cls, c["lang"]), "ctx": c,
                "tasks": [t for t, _ in sections]}

    def next_step(self, c: dict) -> Optional[str]:
        s = self.suggestions.get(c["label"] or "")
        return s[c["lang"]] if s else None

    @staticmethod
    def unsure(cls: Optional[dict], lang: str) -> Optional[str]:
        if not cls or cls["confidence"] >= 0.5:
            return None
        top = sorted(cls["probabilities"].items(), key=lambda kv: -kv[1])[:2]
        a, b = (f"{k} {v:.0%}" for k, v in top)
        return (f"Jeg er ikke helt sikker ({a}, {b}). Fortell meg litt mer, så treffer jeg bedre." if lang == "no"
                else f"I'm not entirely sure ({a}, {b}). Tell me a bit more and I'll land closer.")

    def task_advice(self, text: str, c: dict) -> List[str]:
        names = ", ".join(self.task_triggers)
        return ([f"Jeg kan hjelpe med: {names}. Prøv for eksempel \"{next(iter(self.task_triggers))}\" og et tema."]
                if c["lang"] == "no" else
                [f"I can help with: {names}. Try one of those words along with a topic."])


# ------------------------------------------------------------------ MusicAgent

class MusicAgent(Agent):
    name = "music"
    description = "Blues, jazz, rock and modern music: lyrics, metadata, listening tips and content ideas."
    keywords = {"blues": 3, "jazz": 3, "rock": 2, "guitar": 2, "gitar": 2, "song": 2, "låt": 2, "lyric": 3, "tekst": 1,
                "music": 2, "musikk": 2, "album": 2, "plate": 1, "playlist": 2, "chord": 2, "akkord": 2, "slide": 1,
                "harmonica": 2, "munnspill": 2, "band": 1, "delta": 1, "track": 1, "artist": 1, "spotify": 2, "riff": 2,
                "verse": 1, "vers": 1, "chorus": 1, "refreng": 2, "concert": 2, "konsert": 2, "sing": 1, "synge": 1}
    domains = {"Blues/Roots": 3.0, "American Culture": 0.5}
    task_triggers = {
        "lyrics": ["lyric", "verse", "vers", "stanza", "tekst", "refreng", "analyze", "analyse"],
        "metadata": ["metadata", "tags", "title", "description", "spotify", "release", "tittel", "beskrivelse", "utgivelse"],
        "ideas": ["idea", "idé", "ide", "content", "blog", "blogg", "video", "post", "innhold", "newsletter", "nyhetsbrev"],
        "listen": ["listen", "recommend", "anbefal", "lytte", "playlist", "spilleliste", "hør", "similar", "lignende"],
    }
    suggestions = {
        "Blues/Roots": L("Pick one record and live with it for a week before moving on. Hard Again if you want grit, Live at the Regal if you want to hear a crowd breathe with the band.",
                         "Velg én plate og bo i den en uke før du går videre. Hard Again hvis du vil ha grus, Live at the Regal hvis du vil høre et publikum puste med bandet."),
        "American Culture": L("Tie it to a place: the Delta, the South Side of Chicago, Memphis. Stories stick when they have an address.",
                              "Knytt det til et sted: Deltaen, South Side i Chicago, Memphis. Historier fester seg når de har en adresse."),
        "Business/Monetization": L("Start small. Sell what people already ask you about, a tab pack, a lesson, a signed print, and see what they ask for next.",
                                   "Start smått. Selg det folk allerede spør deg om, et tabulaturhefte, en time, en signert plakat, og se hva de ber om neste gang."),
        "Health/Productivity": L("Twenty minutes a day with a metronome and a rested wrist beats a Sunday marathon every time. Take a break before it hurts, not after.",
                                 "Tjue minutter hver dag med metronom og uthvilt håndledd slår søndagsmaraton hver gang. Ta pause før det gjør vondt, ikke etterpå."),
        "Code/Tech": L("That sounds more like a website or tooling question. The web agent knows more about it: python agents.py --agent web \"...\"",
                       "Dette høres mer ut som et nettside- eller verktøyspørsmål. Web-agenten kan mer om det: python agents.py --agent web \"...\""),
    }

    def task_ideas(self, text: str, c: dict) -> List[str]:
        g, lang, t = GENRES[c["genre"]], c["lang"], c["topic"]
        hooks = [h[lang] for h in pick(g["hooks"], c["seed"], 2)]
        about = f"\"{t}\"" if t else ("temaet" if lang == "no" else "the topic")
        extra = [
            L(f"Write about {about} the way you'd tell it to a friend at the bar: one record, one story, one thing you only noticed on the fifth listen.",
              f"Skriv om {about} slik du ville fortalt det til en venn på puben: én plate, én historie, én ting du først la merke til på femte gjennomlytting."),
            L(f"Film yourself playing {about} on the couch. No editing, no fancy lights. A room that looks lived in is easier to trust.",
              f"Filmer deg selv mens du spiller {about} i sofaen. Ingen klipping, ingen fancy lys. Et rom som ser levd i ut er lettere å stole på."),
            L(f"Make a ten-track playlist around {about} and write one honest sentence per track. Skip the ones you don't love.",
              f"Lag en spilleliste på ti låter rundt {about} og skriv én ærlig setning per låt. Dropp de du ikke er glad i."),
        ]
        out = hooks + [e[lang] for e in pick(extra, c["seed"], 2)]
        out.append("Avslutt hver tekst med én enkel oppfordring: lytt, svar, eller send den videre til noen som trenger den."
                   if lang == "no" else "End each piece with one small ask: listen, reply, or pass it on to someone who needs it.")
        return out

    def task_listen(self, text: str, c: dict) -> List[str]:
        return [x[c["lang"]] for x in pick(GENRES[c["genre"]]["listen"], c["seed"], 3)]

    def task_lyrics(self, text: str, c: dict) -> List[str]:
        lang = c["lang"]
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 3:
            lines = [l.strip() for l in re.split(r"[;/|]", text) if l.strip()]
        if len(lines) < 2:
            return (["Lim inn teksten med én linje per rad (eller skill linjene med ;), så ser jeg på den."]
                    if lang == "no" else ["Paste the lyrics with one line per row (or separate lines with ;) and I'll take a look."])
        norm = [re.sub(r"[^\w\s]", "", l.lower()) for l in lines]
        rep = len(norm) - len(set(norm))
        words = [w for w in tokenize(" ".join(lines), ngrams=False) if w not in STOP]
        ends = [n.split()[-1][-2:] for n in norm if n.split()]
        rhymes = sum(v - 1 for v in Counter(ends).values() if v > 1)
        aab = any(norm[i] == norm[i + 1] != norm[i + 2] for i in range(0, len(norm) - 2, 3))
        top = ", ".join(keywords_of(" ".join(lines), 5))
        if lang == "no":
            out = [f"{len(lines)} linjer og {len(set(words))} ulike ord. Mest brukt: {top}.",
                   f"{rep} gjentatte linjer ({rep / len(lines):.0%}). Det er helt normalt, gjentakelsen er selve poenget i en blues eller en god hook.",
                   f"Ca. {rhymes} linjeslutter som rimer."]
            out.append("Du har en klassisk AAB-strofe: linje to gjentar den første, og tredje linje svarer. Lekkert."
                       if aab else "Prøv AAB: si linjen, si den igjen med litt mer trykk, og la tredje linje svare. Det er der bluesen bor.")
        else:
            out = [f"{len(lines)} lines and {len(set(words))} different words. Most used: {top}.",
                   f"{rep} repeated lines ({rep / len(lines):.0%}). That's normal; repetition is the whole trick of a blues verse or a good hook.",
                   f"Around {rhymes} line endings that rhyme."]
            out.append("That's a proper AAB stanza: line two repeats the first, line three answers it. Nicely done."
                       if aab else "Try AAB: say the line, say it again with a bit more weight, then let line three answer. That's where the blues lives.")
        return out

    def task_metadata(self, text: str, c: dict) -> List[str]:
        lang, g = c["lang"], GENRES[c["genre"]]
        k = keywords_of(text, 5)
        label = (c["label"] or "music").lower().replace("/", " ")
        first = re.split(r"(?<=[.!?])\s", text.strip())[0] if text.strip() else ""
        head = " ".join(k[:3]).title() if k else ("Uten tittel" if lang == "no" else "Untitled")
        desc = (first[:110].rstrip(" .") + ". " if first else "") + g["closing"][lang]
        if lang == "no":
            return [f"Tittelforslag: \"{head}\" eller \"Noen ord om {c['topic'] or 'en gammel plate'}\"",
                    f"Tagger: {', '.join(k + [label])}",
                    f"Beskrivelse (maks 155 tegn): {desc[:155]}",
                    "Husk: artist, tittel, sjanger, toneart/tempo, år, ISRC, coverbilde 3000x3000 og tekstforfatter."]
        return [f"Title ideas: \"{head}\" or \"A Few Words on {(c['topic'] or 'an old record').title()}\"",
                f"Tags: {', '.join(k + [label])}",
                f"Description (max 155 chars): {desc[:155]}",
                "Don't forget: artist, title, genre, key/BPM, year, ISRC, 3000x3000 cover art and songwriter credit."]


# ------------------------------------------------------------------ WebAgent

class WebAgent(Agent):
    name = "web"
    description = "Web projects, SEO, Next.js, Stripe and technical decisions."
    keywords = {"seo": 3, "next.js": 3, "nextjs": 3, "react": 2, "stripe": 3, "website": 2, "web": 1, "html": 1, "css": 1,
                "deploy": 2, "cloudflare": 3, "vercel": 2, "api": 1, "checkout": 2, "payment": 2, "betaling": 2, "landing": 2,
                "domain": 1, "domene": 1, "nettside": 2, "nettbutikk": 2, "javascript": 1, "typescript": 1, "database": 1,
                "hosting": 2, "ssr": 2, "static": 1, "statisk": 1, "docker": 1, "bug": 1, "feil": 0.5}
    domains = {"Code/Tech": 2.5, "Business/Monetization": 1.5}
    task_triggers = {
        "seo": ["seo", "search engine", "søkemotor", "ranking", "meta", "sitemap", "google", "synlig"],
        "stripe": ["stripe", "payment", "checkout", "subscription", "betaling", "abonnement", "kasse"],
        "nextjs": ["next.js", "nextjs", "app router", "ssr", "react"],
        "decision": ["should i", "which", " vs ", "choose", "decision", "hosting", "deploy", "cloudflare", "velge", "bør jeg", "hvilken"],
    }
    suggestions = {
        "Code/Tech": L("Reproduce it in the smallest file you can. Half the time the bug disappears and shows you the cause on the way.",
                       "Gjenskap feilen i den minste filen du klarer. Halvparten av gangene forsvinner bugen, og du ser årsaken på veien."),
        "Business/Monetization": L("Before you tune the price, check that anyone actually clicks \"buy\". Ten real conversations beat a week of A/B testing.",
                                   "Før du justerer prisen: sjekk at noen i det hele tatt trykker \"kjøp\". Ti ekte samtaler slår en uke med A/B-testing."),
        "Blues/Roots": L("For a music site: one page per artist or song, your own photos, your own words. Search engines and people both like that better than a list of links.",
                         "På en musikkside: én side per artist eller låt, egne bilder og egne ord. Både søkemotorer og folk liker det bedre enn en liste med lenker."),
        "American Culture": L("Local, specific content travels well online. A page about one diner or one festival beats a general \"guide to America\".",
                              "Lokalt og konkret innhold sprer seg godt på nett. En side om ett spisested eller én festival slår en generell \"guide til USA\"."),
        "Health/Productivity": L("Ship something small today and stop at a fixed time. Projects survive on rested people, not on all-nighters.",
                                 "Slipp noe lite ut i dag og slutt til et fast klokkeslett. Prosjekter overlever på uthvilte folk, ikke på netter uten søvn."),
    }

    def task_seo(self, text: str, c: dict) -> List[str]:
        lang = c["lang"]
        out = ([
            "Gi hver side sin egen tittel (rundt 50–60 tegn) og sett ordene folk faktisk søker på først.",
            "Én h1 per side, og den sier rett ut hva siden handler om.",
            "Skriv metabeskrivelsen som en kort invitasjon, 120–155 tegn. Den er annonsen din i søkeresultatene.",
            "Bilder trenger ekte alt-tekst: \"mann spiller slide på en veranda\", ikke \"bilde1.jpg\".",
            "Legg inn sitemap.xml og robots.txt, og se i Google Search Console etter en ukes tid.",
            "Fart teller: komprimer bildene, send mindre JavaScript, og sikt på at hovedinnholdet vises på under 2,5 sekunder."]
            if lang == "no" else [
            "Give every page its own title (around 50–60 characters) and put the words people actually search for first.",
            "One h1 per page, and let it say plainly what the page is about.",
            "Write the meta description like a short invitation, 120–155 characters. It's your ad in the search results.",
            "Photos need real alt text: \"man playing slide guitar on a porch\", not \"image1.jpg\".",
            "Add a sitemap.xml and robots.txt, then check Google Search Console after a week or so.",
            "Speed matters: compress images, ship less JavaScript, and aim for the main content showing in under 2.5 seconds."])
        words = tokenize(text, ngrams=False)
        if len(words) >= 40:
            sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
            avg = len(words) / max(1, len(sents))
            kw = ", ".join(keywords_of(text, 5))
            out.append(f"Teksten din: {len(words)} ord, snitt {avg:.0f} ord per setning"
                       f"{' (litt lange, del noen opp)' if avg > 25 else ''}. Ordene som skiller seg ut: {kw}." if lang == "no"
                       else f"Your text: {len(words)} words, about {avg:.0f} words per sentence"
                       f"{' (a bit long, split a few)' if avg > 25 else ''}. The words that stand out: {kw}.")
        return out

    def task_stripe(self, text: str, c: dict) -> List[str]:
        if c["lang"] == "no":
            return ["Start med Stripe Checkout (den hostede siden). Det er mindre å bygge og mindre som kan gå galt.",
                    "Lag Checkout Session på serveren. Den hemmelige nøkkelen skal aldri inn i nettleserkode.",
                    "Fullfør bestillinger fra en verifisert webhook (sjekk signaturen), ikke fra suksess-siden. Folk lukker faner.",
                    "Gjør webhook-håndteringen idempotent: lagre event-id-ene og svar raskt med 2xx.",
                    "Utvikle og test i testmodus med testkort. Bytt nøkler først når du lanserer.",
                    "Abonnement? La Stripe Billing og kundeportalen ta avbestilling og oppgradering, ikke bygg det selv."]
        return ["Start with Stripe Checkout (the hosted page). Less to build, and less that can go wrong.",
                "Create the Checkout Session on the server. The secret key never goes anywhere near browser code.",
                "Fulfil orders from a verified webhook (check the signature), not from the success page. People close tabs.",
                "Make webhook handling idempotent: store the event ids and answer fast with a 2xx.",
                "Build and test in test mode with test cards. Switch keys only at launch.",
                "Subscriptions? Let Stripe Billing and the customer portal handle cancelling and upgrading instead of building it yourself."]

    def task_nextjs(self, text: str, c: dict) -> List[str]:
        if c["lang"] == "no":
            return ["Bruk App Router og Server Components som standard. Legg til 'use client' bare der du trenger state eller effekter.",
                    "Mest fast innhold? Statisk generering eller ISR. Spar SSR til sider som må være personlige per forespørsel.",
                    "Cloudflare Pages: ren statisk eksport (output: 'export') fungerer rett ut av boksen. For SSR og API-ruter trenger du Cloudflare-adapteren, så sjekk hvilke Node-API-er du bruker.",
                    "Hemmeligheter hører hjemme i miljøvariabler på serveren. Bare NEXT_PUBLIC_* når nettleseren.",
                    "next/image, next/font og Metadata API gir deg mye av SEO-grunnmuren gratis."]
        return ["Use the App Router and Server Components by default. Add 'use client' only where you need state or effects.",
                "Mostly fixed content? Static generation or ISR. Keep SSR for pages that really have to be personal per request.",
                "Cloudflare Pages: a plain static export (output: 'export') works straight away. For SSR and API routes you'll need the Cloudflare adapter, so check which Node APIs you rely on.",
                "Secrets live in server-side environment variables. Only NEXT_PUBLIC_* ever reaches the browser.",
                "next/image, next/font and the Metadata API give you most of the SEO groundwork for free."]

    def task_decision(self, text: str, c: dict) -> List[str]:
        low, no = text.lower(), c["lang"] == "no"
        out = (["Tommelregel: velg det enkleste som dekker behovet du har i dag. Du kan bygge ut senere.",
                "Innholdsside eller landingsside: ren statisk HTML eller en statisk generator. Billig, rask og enkelt å legge ut.",
                "App med innlogging og data: Next.js (eller lignende), en administrert database og ferdig innlogging.",
                "Betaling: Stripe Checkout. Du skal ikke lagre kortdata selv."] if no else
               ["Rule of thumb: pick the simplest thing that covers what you need today. You can grow it later.",
                "Content site or landing page: plain static HTML or a static generator. Cheap, fast and easy to host.",
                "App with logins and data: Next.js (or similar), a managed database and hosted sign-in.",
                "Payments: Stripe Checkout. You don't want to store card data yourself."])
        if "cloudflare" in low:
            out.append("Cloudflare Pages passer fint for statisk innhold. Sett output-mappen til 'web'." if no
                       else "Cloudflare Pages is a great fit for static content. Set the output directory to 'web'.")
        if "vercel" in low:
            out.append("Vercel er det enkleste stedet for Next.js med SSR/ISR." if no
                       else "Vercel is the easiest home for Next.js with SSR or ISR.")
        return out


# ------------------------------------------------------------------ registry / router

REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    REGISTRY[cls.name] = cls
    return cls


register(MusicAgent)
register(WebAgent)


class Router:
    """Velger agent (modellens kategori + nøkkelord) og husker de siste meldingene.
    Picks the agent (model category + keywords) and remembers recent interactions."""

    def __init__(self, brain=None, memory_size: int = 5):
        self.brain = brain
        self.agents = {n: c(brain) for n, c in REGISTRY.items()}
        self.memory = Memory(memory_size)

    def route(self, text: str) -> Agent:
        cls = self.brain.predict(text) if self.brain else None
        scores = {n: a.relevance(text, cls) for n, a in self.agents.items()}
        best = max(scores, key=scores.get)
        if scores[best] < 1.0:
            raise LookupError("Fant ingen passende agent / no matching agent. Use --agent or --list.")
        return self.agents[best]

    def run(self, text: str, agent: Optional[str] = None, task: Optional[str] = None) -> dict:
        prev = self.memory.last()
        follow = self.memory.is_followup(text)
        if agent:
            if agent not in self.agents:
                raise KeyError(f"Ukjent agent / unknown agent '{agent}'. Available: {', '.join(self.agents)}")
            chosen = self.agents[agent]
        elif follow and prev:
            # Oppfølger: bli hos forrige agent, med mindre teksten tydelig nevner et annet fagfelt.
            # Follow-up: stay with the last agent unless the text clearly names another one's field.
            chosen = self.agents[prev["agent"]]
            for a in self.agents.values():
                if a.name != chosen.name and a.keyword_score(text) >= 2:
                    chosen = a
        else:
            chosen = self.route(text)
        res = chosen.run(text, task, prev, follow)
        res["followup"] = bool(follow and prev)
        c = res["ctx"]
        self.memory.add(text=text, agent=chosen.name, tasks=res["tasks"], topic=c["topic"], genre=c["genre"],
                        label=c["label"], lang=c["lang"])
        return res


def load_brain(path: str):
    try:
        from predict import Brain
        return Brain(path)
    except Exception as e:  # mangler modell/torch -> agentene virker med nøkkelord / still work on keywords
        say(f"Merk: kjører uten modell ({e}). Kun nøkkelord brukes.",
            f"Note: running without model ({e}). Keywords only.")
        return None


def show(res: dict) -> None:
    c = res["classification"]
    cls = f"  [{c['label']} {c['confidence']:.0%}]" if c else ""
    print(f"\n== {res['agent']} ({res['lang']}){cls}{' (oppfølging/follow-up)' if res.get('followup') else ''}")
    if res["unsure"]:
        print(f"  ? {res['unsure']}")
    for task, lines in res["sections"]:
        print(f"  [{task}]")
        for line in lines:
            print(f"    - {line}")
    if res["next"]:
        print(f"  {'Neste steg' if res['lang'] == 'no' else 'Next step'}:\n    {res['next']}")


def selftest() -> int:
    """Kjør alle oppgaver på norsk og engelsk og se etter typisk AI-språk. / Style check for stock AI phrasing."""
    router = Router(None)
    probes = ["blues slide guitar ideas", "idéer til en bluesblogg om slide", "jazz listen recommend", "anbefal rock å lytte til",
              "I woke up this morning; I woke up this morning; blues walking like a man", "metadata for en jazzlåt",
              "seo checklist", "seo for nettsiden min", "stripe checkout", "stripe betaling", "next.js app router", "next.js på norsk",
              "should i use cloudflare or vercel", "hvilken hosting bør jeg velge cloudflare"]
    bad = 0
    for q in probes:
        r = router.run(q, agent="web" if any(k in q for k in ("seo", "stripe", "next", "hosting", "cloudflare")) else "music")
        blob = " ".join(l for _, ls in r["sections"] for l in ls).lower()
        blob += " " + " ".join(t[lang].lower() for g in GENRES.values() for key in ("listen", "hooks") for t in g[key] for lang in ("en", "no"))
        for b in BANNED:
            if b in blob:
                print(f"STIL/STYLE: '{b}' in output for: {q}")
                bad += 1
    print("Stiltest OK / style test OK" if not bad else f"{bad} treff / hits")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="KM-TECH BRAIN agents (lokalt / local)")
    p.add_argument("text", nargs="*")
    p.add_argument("--agent", choices=sorted(REGISTRY), help="Velg agent / force an agent")
    p.add_argument("--task", help="Velg oppgave / force a task (ideas, lyrics, metadata, listen, seo, stripe, nextjs, decision)")
    p.add_argument("--model", default="models/brain.pth")
    p.add_argument("--list", action="store_true")
    p.add_argument("--interactive", action="store_true", help="Samtale med minne / conversation with memory")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()
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
        say("Skriv en oppgave. /historikk viser minnet, /glem tømmer det, tom linje avslutter.",
            "Type a request. /history shows memory, /reset clears it, empty line quits.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            if line in ("/history", "/historikk"):
                for m in router.memory.items:
                    print(f"  {m['agent']}/{','.join(m['tasks'])}: {m['text'][:70]}")
            elif line in ("/reset", "/glem"):
                router.memory.clear()
                print("  (tømt / cleared)")
            else:
                go(line)
        return 0
    if not a.text:
        p.print_help()
        return 0
    go(" ".join(a.text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
