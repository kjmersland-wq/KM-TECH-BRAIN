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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def has_word(low: str, kw: str) -> bool:
    """Treff på ordstart, så 'ide' ikke matcher 'slide' men 'ideer' matcher 'ide'.
    Word-start match: 'ide' must not hit inside 'slide', but 'ideer' still matches 'ide'."""
    return re.search(r"(?<!\w)" + re.escape(kw.strip()), low) is not None


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
        starts = ("more", "another", "and ", "also", "what about", "how about", "flere", "mer ", "og ", "make it", "gjør den", "gjør det", "now ", "change ", "endre ", "legg til", "add ",
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

GENRES["blues"]["listen"] += [
    L("Robert Johnson, \"King of the Delta Blues Singers\" (1961). The compilation that introduced him to a whole new generation, hiss and all.",
      "Robert Johnson, \"King of the Delta Blues Singers\" (1961). Samleplata som introduserte ham for en helt ny generasjon, susing og alt."),
]
GENRES["jazz"]["listen"] += [
    L("Art Blakey & the Jazz Messengers, \"Moanin'\" (1958). Hard bop with a bounce. You'll be nodding before the horns come in.",
      "Art Blakey & the Jazz Messengers, \"Moanin'\" (1958). Hard bop med sprett i. Du nikker med før blåserne er inne."),
    L("Thelonious Monk, \"Misterioso\" (1958), recorded live at the Five Spot. Crooked, funny and completely sure of itself.",
      "Thelonious Monk, \"Misterioso\" (1958), live fra Five Spot. Skjev, morsom og helt sikker på seg selv."),
]
GENRES["rock"]["listen"] += [
    L("The Who, \"Live at Leeds\" (1970). Loud, sloppy and thrilling. Exactly what a rock show should be.",
      "The Who, \"Live at Leeds\" (1970). Høyt, slurvete og spennende. Akkurat sånn en rockekonsert skal være."),
    L("Chuck Berry, \"Chuck Berry Is on Top\" (1959). Every rock guitarist has stolen from this guy, and he knew it.",
      "Chuck Berry, \"Chuck Berry Is on Top\" (1959). Hver eneste rockegitarist har stjålet fra denne fyren, og han visste det."),
]
GENRES["blues"]["intro"] = L(
    "Blues is the root system under most of what you hear today. It grew out of African American communities in the Deep South in the late 1800s, built on call and response and those blue notes that sit in the cracks between the piano keys. The twelve-bar form sounds simple until you try to make it mean something. Start with Skip James, jump to Muddy Waters in Chicago, and you'll hear the same ache from two different rooms.",
    "Bluesen er rotsystemet under det meste du hører i dag. Den vokste fram i afroamerikanske miljøer i sørstatene på slutten av 1800-tallet, bygget på kall og svar og de såkalte bluesnotene som ligger i sprekken mellom tangentene på pianoet. Tolvtaktsformen høres enkel ut helt til du prøver å få den til å bety noe. Start med Skip James, hopp så til Muddy Waters i Chicago, så hører du den samme smerten fra to forskjellige rom.")
GENRES["jazz"]["intro"] = L(
    "Jazz started in New Orleans in the early 1900s and has never sat still since. The heart of it is improvisation: musicians making it up together, listening hard, answering each other. Swing, bebop, cool, free, every decade took it somewhere new. If you've never listened properly, give Kind of Blue one evening with nothing else going on.",
    "Jazzen startet i New Orleans tidlig på 1900-tallet og har aldri satt seg ned siden. Kjernen er improvisasjon: musikere som finner det opp sammen, lytter nøye og svarer hverandre. Swing, bebop, cool, free, hvert tiår tok den et nytt sted. Har du aldri hørt skikkelig etter, sett på Kind of Blue en kveld uten at noe annet pågår.")
GENRES["rock"]["intro"] = L(
    "Rock and roll came out of rhythm and blues and country in the American South in the early fifties, with Chuck Berry's guitar and Little Richard's piano doing the heavy lifting. Then the electric guitar got louder every year: Hendrix, Zeppelin, Sabbath. Nearly every great rock riff is a blues phrase in a leather jacket.",
    "Rocken vokste ut av rhythm and blues og country i sørstatene tidlig på femtitallet, med Chuck Berrys gitar og Little Richards piano som de tunge løftene. Så ble den elektriske gitaren høyere for hvert år: Hendrix, Zeppelin, Sabbath. Nesten hvert eneste store rockeriff er en bluesfrase i skinnjakke.")
GENRES["blues"]["next"] = L(
    "Pick one record and live with it for a week before moving on. Hard Again if you want grit, Live at the Regal if you want to hear a crowd breathe with the band.",
    "Velg én plate og bo i den en uke før du går videre. Hard Again hvis du vil ha grus, Live at the Regal hvis du vil høre et publikum puste med bandet.")
GENRES["jazz"]["next"] = L(
    "Put on one album from start to finish with the phone in another room. Then look up who played on it. Jazz is a family tree, and every name leads to three more records.",
    "Sett på én plate fra start til slutt med telefonen i et annet rom. Slå så opp hvem som spilte på den. Jazz er et slektstre, og hvert navn leder til tre nye plater.")
GENRES["rock"]["next"] = L(
    "Find the riff you love and learn it by ear before you look up the tab. You'll remember it for good, and you'll hear what the guitarist actually did.",
    "Finn riffet du er mest glad i og lær det på gehør før du sjekker tabulaturen. Da husker du det for alltid, og du hører hva gitaristen faktisk gjorde.")
GENRES["any"]["next"] = L(
    "Tell me a genre or an artist you like, and I'll point you somewhere good.",
    "Si meg en sjanger eller en artist du liker, så peker jeg deg et godt sted.")
GENRES["any"]["intro"] = L(
    "Tell me which genre you're curious about, blues, jazz or rock, and I'll give you a proper introduction.",
    "Si hvilken sjanger du er nysgjerrig på, blues, jazz eller rock, så gir jeg deg en ordentlig introduksjon.")

CRAFT_TIPS = [  # (nøkkelord, tips): konkrete håndverksråd som slår inn før de generelle / specific craft advice beats the generic
    (["harmonica", "munnspill", "harp", "harpe"],
     L("For a blues in G, grab a C harp and play it in second position, cross harp. Bend by changing the shape of your mouth, not by blowing harder.",
       "Til en blues i G tar du en C-harpe og spiller i andre posisjon, cross harp. Bend ved å endre formen på munnen, ikke ved å blåse hardere.")),
    (["slide", "bottleneck"],
     L("Keep the slide light on the string, right over the fret wire, and mute behind it with your other fingers. Intonation first, vibrato later.",
       "Hold slidestålet lett mot strengen, rett over båndet, og demp bak det med de andre fingrene. Først tonetreff, så vibrato.")),
    (["open g", "open d", "åpen g", "åpen d", "tuning", "stemming", "stemme gitar", "stemmer"],
     L("Open G is D-G-D-G-B-D and open D is D-A-D-F#-A-D. Same shapes, different mood. Try both on one song and keep the one that sounds like you.",
       "Åpen G er D-G-D-G-B-D og åpen D er D-A-D-F#-A-D. Samme grep, ulik stemning. Prøv begge på én låt og behold den som høres ut som deg.")),
    (["riff", "distortion", "forvrengning", "overdrive", "amp ", "forsterker"],
     L("Turn the gain down until you can hear every note, then bring it back up a notch. Most great riff tones are cleaner than you think.",
       "Skru ned gainen til du hører hver tone, og skru den opp igjen en hakk. De fleste store riff-lydene er renere enn du tror.")),
    (["improvis", "solo"],
     L("Play less. Sing the line in your head first, then find it on the neck. The space between the notes is half the solo.",
       "Spill mindre. Syng frasen i hodet først, og finn den så på gripebrettet. Pausene mellom tonene er halve soloen.")),
    (["lyric", "tekst", "verse", "vers", "refreng", "chorus"],
     L("Say the line out loud before you write it down. If you trip over it, so will the singer.",
       "Si linjen høyt før du skriver den ned. Snubler du i den, gjør sangeren det også.")),
]


def craft_tip(text: str, lang: str) -> Optional[str]:
    low = text.lower()
    for kws, tip in CRAFT_TIPS:
        if any(has_word(low, k) for k in kws):
            return tip[lang]
    return None


WEB_HARD = ["seo", "stripe", "next.js", "nextjs", "checkout", "deploy", "cloudflare", "vercel", "nettside", "website",
            "webshop", "nettbutikk", "hosting", "docker", "api", "css", "html", "react", "typescript", "javascript",
            "domene", "domain", "sitemap", "wordpress", "webhook", "frontend", "backend"]

LABEL_GENRE = {"Blues": "blues", "Jazz": "jazz", "Rock": "rock", "Blues/Roots": "blues"}  # modell-/biblioteklabel -> sjanger


def genre_by_keywords(text: str) -> Optional[str]:
    low = text.lower()
    scores = {g: sum(has_word(low, k) for k in d["kw"]) for g, d in GENRES.items() if g != "any"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


BANNED = ["delve", "unlock", "journey", "tapestry", "game-changer", "game changer", "elevate", "in today's", "dive into",
          "seamless", "leverage", "unleash", "embark", "landscape", "realm", "furthermore", "moreover", "in conclusion",
          "ultimate guide", "whether you're", "i dagens", "utforsk", "spennende reise", "neste nivå", "frigjør",
          "dykk ned", "ta steget", "hos oss finner du"]


def detect_genre(text: str, label: Optional[str] = None, prev: Optional[str] = None) -> str:
    low = text.lower()
    scores = {g: sum(has_word(low, k) for k in d["kw"]) for g, d in GENRES.items() if g != "any"}
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
    default_task: Optional[str] = None  # brukes når ingen oppgave nevnes / used when no task is named

    def __init__(self, brain=None, retriever=None, genre_brain=None):
        self.brain = brain              # kategori-modell (models/brain.pth) / category model
        self.retriever = retriever      # retrieve.Retriever over biblioteket ditt / over your library
        self.genre_brain = genre_brain  # valgfri sjanger-modell (models/genres.pth) / optional genre model

    def related(self, text: str, c: dict) -> List[dict]:
        """De mest like eksemplene i biblioteket ditt. / Most similar examples from your library."""
        if not self.retriever:
            return []
        return self.retriever.similar(text, k=3, min_score=0.15)

    def task_related(self, text: str, c: dict) -> List[str]:
        hits = self.related(text, c)
        if not hits:
            return (["Fant ingenting som ligner i biblioteket ditt ennå. Legg til flere eksempler med python ingest.py."]
                    if c["lang"] == "no" else ["Nothing close in your library yet. Add more examples with python ingest.py."])
        return [f"[{h['label']}] {h['text']}" for h in hits]

    def keyword_score(self, text: str) -> float:
        low = text.lower()
        return sum(w for k, w in self.keywords.items() if has_word(low, k))

    def relevance(self, text: str, cls: Optional[dict] = None) -> float:
        score = self.keyword_score(text)
        if cls and "label" in cls:
            score += self.domains.get(cls["label"], 0.0) * cls["confidence"] * 2
        return score

    def pick_tasks(self, text: str, limit: int = 3) -> List[str]:
        low = text.lower()
        hits = {t: sum(has_word(low, w) for w in ws) for t, ws in self.task_triggers.items()}
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
        if not sections and self.default_task and c.get("genre", "any") != "any":
            sections.append((self.default_task, getattr(self, f"task_{self.default_task}")(text, c)))
        if not sections:
            sections.append(("advice", self.task_advice(text, c)))
        return {"agent": self.name, "lang": c["lang"], "classification": cls, "sections": sections,
                "next": self.next_step(c), "unsure": self.unsure(cls, c["lang"]), "ctx": c,
                "related": self.related(text, c),
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
        "listen": ["listen", "recommend", "anbefal", "lytte", "playlist", "spilleliste", "hør"],
        "related": ["related", "similar", "lignende", "liknende", "more like", "mer som", "find examples", "finn eksempler"],
        "explain": ["explain", "what is", "hva er", "forklar", "history", "historie", "intro", "guide", "introduction"],
    }

    def context(self, text: str, cls: Optional[dict], prev: Optional[dict], follow: bool) -> dict:
        """Sjanger: nøkkelord først, så sjangermodell, så søk i biblioteket (kNN). Lets in that order."""
        c = super().context(text, cls, prev, follow)
        c["genre_how"] = "keywords" if genre_by_keywords(text) else None
        if c["genre_how"] or (follow and prev and len(text.split()) <= 3):
            return c
        if self.genre_brain:
            g = self.genre_brain.predict(text)
            if g["confidence"] >= 0.5 and g["label"] in LABEL_GENRE:
                c["genre"], c["genre_how"] = LABEL_GENRE[g["label"]], f"genre model {g['confidence']:.0%}"
                return c
        if self.retriever:
            label, share = self.retriever.vote(text)
            if label in LABEL_GENRE and share >= 0.5:
                c["genre"], c["genre_how"] = LABEL_GENRE[label], f"library {share:.0%}"
        return c

    def relevance(self, text: str, cls: Optional[dict] = None) -> float:
        """Sjangerord (jazz, Hendrix, Muddy ...) og biblioteket teller tungt, slik at musikk ikke
        havner hos WebAgent bare fordi kategorimodellen er usikker. / Genre words and the library count heavily."""
        low = text.lower()
        hard = any(has_word(low, w) for w in WEB_HARD)
        # Med tekniske nøkkelord er kategorimodellens musikk-hint (Blues/Roots) ikke avgjørende
        # With technical terms the model's music hint (Blues/Roots) must not decide it
        score = self.keyword_score(text)
        if cls and "label" in cls and not hard:
            score += self.domains.get(cls["label"], 0.0) * cls["confidence"] * 2
        if genre_by_keywords(text):
            # 'SEO for my blues blog' er et web-spørsmål om musikk, ikke et musikkspørsmål
            # 'SEO for my blues blog' is a web question about music, not a music question
            score += 1 if hard else 5
        elif self.retriever and not hard:
            label, share = self.retriever.vote(text)
            if label in LABEL_GENRE and share >= 0.6:
                score += 2
        return score

    default_task = "listen"

    def next_step(self, c: dict) -> Optional[str]:
        """Sjangeren styrer, men sterke signaler om penger/helse (f.eks. 'selge', 'vondt i hånda') får egne råd.
        Genre drives it, but a confident money/health signal gets its own advice."""
        conf = c["cls"]["confidence"] if c["cls"] else 0.0
        tip = craft_tip(c["seed"], c["lang"])
        if tip:
            return tip
        if c["label"] in ("Business/Monetization", "Health/Productivity") and conf >= 0.7:
            return self.suggestions[c["label"]][c["lang"]]
        if c["genre"] != "any":
            return GENRES[c["genre"]]["next"][c["lang"]]
        s = self.suggestions.get(c["label"] or "")
        return s[c["lang"]] if s else None

    def task_explain(self, text: str, c: dict) -> List[str]:
        return [GENRES[c["genre"]]["intro"][c["lang"]]]
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

WEB_NEXT = {
    "SEO": L("Pick one page and fix just that one: a title that says what it is, a single h1, and a meta description someone would actually click. Then leave it alone for two weeks.",
             "Velg én side og fiks bare den: en tittel som sier hva siden er, én h1 og en metabeskrivelse noen faktisk vil klikke på. La den stå i fred i to uker etterpå."),
    "Frontend": L("Get it working in the smallest component you can, then open it in a second browser. Most layout bugs show up the moment you do.",
                  "Få det til å fungere i den minste komponenten du klarer, og åpne den i en annen nettleser. De fleste layoutfeil dukker opp med en gang."),
    "Backend": L("Put a log line before and after the step that fails and reproduce it once. Nine times out of ten the log tells you more than the debugger does.",
                 "Legg en loggmelding før og etter steget som feiler, og gjenskap feilen én gang. Ni av ti ganger forteller loggen deg mer enn debuggeren."),
    "Payments": L("Run the whole flow once in test mode, including a failed payment, and look at what your webhook actually receives. That's where the surprises are.",
                  "Kjør hele flyten én gang i testmodus, også en mislykket betaling, og se hva webhooken din faktisk mottar. Det er der overraskelsene sitter."),
    "Hosting": L("Read the build log from the top, not the bottom. The real error is usually the first red line, and everything after it is fallout.",
                 "Les byggloggen fra toppen, ikke bunnen. Den egentlige feilen er som regel den første røde linjen, og alt etter den er følgefeil."),
}

WEB_TOPIC_KW = {"SEO": ["seo", "sitemap", "meta description", "metabeskrivelse", "google search", "backlink", "søkemotor"],
                "Payments": ["stripe", "checkout", "payment", "betaling", "vipps", "subscription", "abonnement", "webhook"],
                "Hosting": ["deploy", "cloudflare", "vercel", "hosting", "docker", "dns", "domene", "domain", "github actions", "ci/cd"],
                "Frontend": ["react", "css", "tailwind", "next.js", "nextjs", "html", "frontend", "component", "komponent"],
                "Backend": ["api", "database", "postgres", "backend", "fastapi", "express", "prisma", "server"]}


def web_topic_by_keywords(text: str) -> Optional[str]:
    low = text.lower()
    scores = {t: sum(has_word(low, k) for k in ks) for t, ks in WEB_TOPIC_KW.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


class WebAgent(Agent):
    name = "web"
    description = "Web projects, SEO, Next.js, Stripe and technical decisions."
    keywords = {"seo": 3, "next.js": 3, "nextjs": 3, "react": 2, "stripe": 3, "website": 3, "webshop": 3, "web": 1, "html": 1, "css": 1,
                "deploy": 2, "cloudflare": 3, "vercel": 2, "api": 1, "checkout": 2, "payment": 2, "betaling": 2, "landing": 2,
                "domain": 1, "domene": 1, "nettside": 3, "nettbutikk": 2, "webhook": 2, "frontend": 2, "backend": 2, "sitemap": 3, "javascript": 1, "typescript": 1, "database": 1,
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

    task_triggers["related"] = ["related", "similar", "lignende", "liknende", "more like", "mer som", "find examples", "finn eksempler"]

    def relevance(self, text: str, cls: Optional[dict] = None) -> float:
        """Tekniske nøkkelord (SEO, Stripe, deploy ...) er et sterkt signal. Kategorimodellen alene er det ikke:
        uten nøkkelord teller den bare halvt. / Technical terms are a strong signal; the category model alone is a weak one."""
        kw = self.keyword_score(text)
        score = kw
        if cls and "label" in cls:
            score += self.domains.get(cls["label"], 0.0) * cls["confidence"] * 2 * (1.0 if kw else 0.5)
        low = text.lower()
        if any(has_word(low, w) for w in WEB_HARD):
            score += 3
        return score

    def context(self, text: str, cls: Optional[dict], prev: Optional[dict], follow: bool) -> dict:
        """Tema: nøkkelord først, så avstemning blant de mest like eksemplene. / Topic: keywords, then library vote."""
        c = super().context(text, cls, prev, follow)
        topic, how = web_topic_by_keywords(text), "keywords"
        if not topic and self.retriever and not (follow and prev and len(text.split()) <= 3):
            label, share = self.retriever.vote(text)
            if label in WEB_NEXT and share >= 0.5:
                topic, how = label, f"library {share:.0%}"
        if not topic and follow and prev:
            topic, how = prev.get("web_topic"), "memory"
        c["web_topic"], c["topic_how"] = topic, (how if topic else None)
        return c

    def next_step(self, c: dict) -> Optional[str]:
        if c.get("web_topic"):
            return WEB_NEXT[c["web_topic"]][c["lang"]]
        return super().next_step(c)

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

# ------------------------------------------------------------------ CodeAgent (lokal LLM / local LLM)

CODE_PHRASES = [  # tydelige bestillinger av kode / clear requests for code
    r"\bwrite (me )?(a|an|some)\b.{0,25}\b(function|script|class|component|api|route|endpoint|test|query|program|snippet|regex|hook|middleware|code)\b",
    r"\b(write|generate|give me) (the )?code\b", r"\bcode example\b", r"\bimplement\b", r"\brefactor\b", r"\bunit tests? for\b",
    r"\bskriv (meg )?(en|et|litt)\b.{0,25}\b(funksjon|script|klasse|komponent|rute|spørring|test|program|kode|regex)\b",
    r"\b(lag|generer) (en|et|litt)\b.{0,25}\b(funksjon|script|klasse|komponent|rute|kode)\b", r"\bkodeeksempel\b", r"\bimplementer\b",
]


def split_answer(text: str) -> Tuple[str, str]:
    """Del LLM-svaret i (kode, forklaring). Kodeblokkene beholdes uendret. / Split into (code, explanation)."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    explanation = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.S).strip()
    explanation = re.sub(r"^(\*\*)?(explanation|forklaring)\s*:?(\*\*)?\s*:?\s*", "", explanation, flags=re.I)  # modellen skriver ofte selv «Explanation:»
    if not blocks:  # modellen glemte kodeblokk-formatet / the model skipped the fenced format
        return text.strip(), ""
    return "\n\n".join(b.rstrip() for b in blocks), explanation


class CodeAgent(Agent):
    """Enkel kodehjelper på en lokal LLM (Ollama). Ingen filendringer: du får kode og en kort forklaring.
    Basic coding helper on a local LLM (Ollama). It never touches your files; you get code plus a short explanation."""
    name = "code"
    description = "Code generation with a local LLM through Ollama (python llm.py --models)."
    task_triggers = {"generate": ["write", "implement", "skriv", "lag", "kode", "code"]}

    SYSTEM = ("You are a careful senior software engineer helping on small projects. Reply in exactly this shape: "
              "one fenced code block with the complete, runnable code (put the language name after the opening backticks), "
              "then a short explanation of at most five sentences. Mention assumptions and anything that must be installed. "
              "No filler such as 'Certainly' or 'Here is'. Do not invent library functions; if you are unsure about an API, "
              "say so in the explanation. Write the explanation like a friendly colleague: plain, specific, natural. {lang}")

    def __init__(self, brain=None, retriever=None, genre_brain=None):
        super().__init__(brain, retriever, genre_brain)
        self.history: List[dict] = []   # siste utveksling, så 'gjør den om til TypeScript' virker / last exchange for follow-ups
        self._up: Optional[bool] = None
        self.model_override: Optional[str] = None

    def available(self) -> bool:
        if self._up is None:
            import llm
            self._up = llm.is_running()
        return self._up

    def relevance(self, text: str, cls: Optional[dict] = None) -> float:
        """Bare med tydelig kodebestilling, og bare hvis Ollama svarer. / Only for a clear code request and only if Ollama is up."""
        low = text.lower()
        if not any(re.search(p, low) for p in CODE_PHRASES):
            return 0.0
        return 12.0 if self.available() else 0.0

    def run(self, text: str, task: Optional[str] = None, prev: Optional[dict] = None, follow: bool = False) -> dict:
        import llm
        lang = prev["lang"] if (follow and prev and len(text.split()) <= 3) else lang_of(text)
        c = {"lang": lang, "topic": topic_of(text), "genre": "any", "label": None, "cls": None, "seed": text,
             "web_topic": None, "genre_how": None, "topic_how": None}
        res = {"agent": self.name, "lang": lang, "classification": None, "next": None, "unsure": None, "ctx": c,
               "related": [], "tasks": ["generate"], "code": "", "explanation": "", "model": None}
        try:
            name, warn = llm.pick_model(llm.installed_models(), self.model_override)
            sys_prompt = self.SYSTEM.format(lang="Write the explanation in natural Norwegian (bokmål)." if lang == "no"
                                            else "Write the explanation in English.")
            msgs = [{"role": "system", "content": sys_prompt}] + self.history[-4:] + [{"role": "user", "content": text}]
            answer = llm.chat(msgs, name)
        except llm.OllamaError as e:
            res["sections"] = [("error", [str(e)])]
            res["explanation"] = str(e)
            return res
        code, expl = split_answer(answer)
        self.history = (self.history + [{"role": "user", "content": text}, {"role": "assistant", "content": answer}])[-4:]
        res.update(code=code, explanation=expl, model=name, warning=warn, sections=[("code", [code]), ("explanation", [expl])])
        return res


REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    REGISTRY[cls.name] = cls
    return cls


register(MusicAgent)
register(WebAgent)
register(CodeAgent)


class Router:
    """Velger agent (modellens kategori + nøkkelord) og husker de siste meldingene.
    Picks the agent (model category + keywords) and remembers recent interactions."""

    def __init__(self, brain=None, memory_size: int = 5, retriever=None, genre_brain=None, retrievers=None):
        self.brain = brain
        libs = dict(retrievers or {})
        if retriever is not None:
            libs.setdefault("music", retriever)  # eldre kall / older call style
        self.agents = {n: c(brain, libs.get(n), genre_brain if n == "music" else None) for n, c in REGISTRY.items()}
        self.memory = Memory(memory_size)

    def route(self, text: str) -> Agent:
        cls = self.brain.predict(text) if self.brain else None
        scores = {n: a.relevance(text, cls) for n, a in self.agents.items()}
        best = max(scores, key=scores.get)
        if scores[best] < 1.0 and any(re.search(p, text.lower()) for p in CODE_PHRASES):
            raise LookupError("Dette ser ut som en kodebestilling, men Ollama svarer ikke. Start Ollama (se: python llm.py --models). "
                              "[EN] This looks like a code request, but Ollama isn't responding. Start Ollama (see: python llm.py --models).")
        if scores[best] < 1.0:  # for svakt signal -> ikke gjett / too weak a signal, don't guess
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
                        label=c["label"], lang=c["lang"], web_topic=c.get("web_topic"))
        return res


def load_brain(path: str):
    try:
        from predict import Brain
        return Brain(path)
    except Exception as e:  # mangler modell/torch -> agentene virker med nøkkelord / still work on keywords
        say(f"Merk: kjører uten modell ({e}). Kun nøkkelord brukes.",
            f"Note: running without model ({e}). Keywords only.")
        return None


def load_library(path: str):
    """Valgfritt søkebibliotek. Mangler det, virker agentene uten. / Optional; agents work without it."""
    if not path or not Path(path).exists():
        return None
    try:
        from retrieve import Retriever
        return Retriever(path, quiet=True)
    except Exception as e:
        say(f"Merk: biblioteket ble ikke lastet ({e}).", f"Note: library not loaded ({e}).")
        return None


def load_genre_model(path: str):
    if not path or not Path(path).exists():
        return None
    try:
        from predict import Brain
        return Brain(path)
    except Exception as e:
        say(f"Merk: sjangermodellen ble ikke lastet ({e}).", f"Note: genre model not loaded ({e}).")
        return None


def show_code(res: dict) -> None:
    """Kode vises urørt (ikke som punktliste), så den kan kopieres rett ut. / Code is printed untouched so you can copy it."""
    no = res["lang"] == "no"
    if res["code"]:
        print(f"\n== code ({res['model']})" + (f"\n  ! {res['warning']}" if res.get("warning") else ""))
        print("\n" + res["code"] + "\n")
        if res["explanation"]:
            print(f"{'Forklaring' if no else 'Explanation'}: {res['explanation']}")
        print(f"\n{'Ingen filer er endret. Les gjennom og kjør koden selv; lokale modeller kan finne på funksjoner som ikke finnes.' if no else 'No files were changed. Read it and run it yourself; local models can invent functions that do not exist.'}")
    else:
        print(f"\n{res['explanation']}")


def show(res: dict) -> None:
    if res["agent"] == "code":
        return show_code(res)
    c = res["classification"]
    # MusicAgent: sjangeren er hovedsignalet, den grove kategorimodellen vises bare når sjangeren er ukjent
    # MusicAgent: the genre is the main signal; the coarse category only shows when the genre is unknown
    hide = res["agent"] == "music" and res["ctx"]["genre"] != "any"
    cls = f"  [{c['label']} {c['confidence']:.0%}]" if c and not hide else ""
    print(f"\n== {res['agent']} ({res['lang']}){cls}{' (oppfølging/follow-up)' if res.get('followup') else ''}")
    if res["unsure"]:
        print(f"  ? {res['unsure']}")
    how = res["ctx"].get("genre_how")
    if res["agent"] == "music" and res["ctx"]["genre"] != "any":
        print(f"  {'sjanger' if res['lang'] == 'no' else 'genre'}: {res['ctx']['genre']} ({how or 'memory'})")
    if res["agent"] == "web" and res["ctx"].get("web_topic"):
        print(f"  {'tema' if res['lang'] == 'no' else 'topic'}: {res['ctx']['web_topic']} ({res['ctx']['topic_how']})")
    for task, lines in res["sections"]:
        print(f"  [{task}]")
        for line in lines:
            print(f"    - {line}")
    if res.get("related") and "related" not in res["tasks"]:
        print(f"  {'Fra biblioteket ditt' if res['lang'] == 'no' else 'From your library'}:")
        for h in res["related"]:
            print(f"    ~ {h['score']:.2f} [{h['label']}] {h['text'][:90]}")
    if res["next"]:
        print(f"  {'Neste steg' if res['lang'] == 'no' else 'Next step'}:\n    {res['next']}")


def build_router(model: str = "models/brain.pth", music_library: str = "data/music_genres.csv",
                 web_library: str = "data/web_topics.csv", genre_model: str = "models/genres.pth", quiet: bool = False) -> Router:
    """Sett sammen alt som finnes lokalt; det som mangler hoppes over. / Assemble whatever exists locally."""
    brain = load_brain(model) if Path(model).exists() or not quiet else None
    return Router(brain, retrievers={"music": load_library(music_library), "web": load_library(web_library)},
                  genre_brain=load_genre_model(genre_model))


def brief(res: dict) -> None:
    """Kort svar: kategori/sjanger, lignende fra dine data, og én naturlig tanke.
    Short answer: category/genre, similar examples from your data, and one natural suggestion."""
    if res["agent"] == "code":
        return show_code(res)
    no, c = res["lang"] == "no", res["ctx"]
    if res["agent"] == "music" and c["genre"] != "any":
        head = f"{'Sjanger' if no else 'Genre'}: {c['genre']} ({c.get('genre_how') or 'minne/memory'})"
    elif res["agent"] == "web" and c.get("web_topic"):
        head = f"{'Tema' if no else 'Topic'}: {c['web_topic']} ({c['topic_how']})"
    elif res["classification"]:
        k = res["classification"]
        head = f"{'Kategori' if no else 'Category'}: {k['label']} ({k['confidence']:.0%})"
    else:
        head = f"{'Agent' if no else 'Agent'}: {res['agent']}"
    print(f"\n  {head}  ·  agent: {res['agent']}")
    if res["unsure"]:
        print(f"  ? {res['unsure']}")
    if res.get("related"):
        print(f"  {'Likt i dataene dine' if no else 'Close to this in your data'}:")
        for h in res["related"]:
            print(f"    {h['score']:.2f} [{h['label']}] {h['text'][:95]}")
    elif res["agent"] in ("music", "web"):
        print(f"  {'Ingenting likt i dataene dine ennå.' if no else 'Nothing close in your data yet.'}")
    if res["next"]:
        print(f"\n  {'Min tanke' if no else 'My take'}: {res['next']}")
    if res["agent"] == "music" and c["genre"] != "any":
        tip = pick(GENRES[c["genre"]]["listen"], c["seed"], 1)[0][res["lang"]]
        print(f"  {'Lytt til' if no else 'Listen to'}: {tip}")
    print(f"\n  ({'skriv «mer» for hele svaret' if no else 'type \"more\" for the full answer'})")


# Rutingtest: (tekst, forventet agent). Fanger feil som 'SEO for bluesbloggen' -> musikk.
ROUTING_PROBES = [
    ("ideas for a blues blog", "music"), ("Hva er jazz egentlig?", "music"), ("recommend some rock to listen to", "music"),
    ("slide i åpen G på en gammel resonator", "music"), ("Stones' basement record, anything like it?", "music"),
    ("lyrics: I woke up this morning, my baby was gone", "music"), ("metadata for my new jazz song", "music"),
    ("Hendrix vs Van Halen tapping", "music"), ("playlist for a rainy Sunday, jazz", "music"),
    ("Spotify release checklist for my band", "music"), ("anbefal noe blues å lytte til", "music"),
    ("SEO checklist for my website", "web"), ("stripe checkout for my webshop", "web"), ("Next.js app router vs pages", "web"),
    ("hvordan deployer jeg på Cloudflare Pages", "web"), ("CORS error from my FastAPI backend", "web"),
    ("vercel or cloudflare for hosting", "web"), ("Docker container exits immediately", "web"),
    ("nettbutikk med Stripe og Next.js", "web"), ("SEO for my blues blog", "web"), ("sell my guitar lessons with Stripe", "web"),
    ("build a website for my jazz band", "web"), ("bandet vårt trenger en nettside", "web"), ("SEO for bluesbloggen min", "web"),
]


def routing_check(router: Router) -> int:
    bad = 0
    for text, want in ROUTING_PROBES:
        try:
            got = router.route(text).name
        except LookupError:
            got = "none"
        if got != want:
            print(f"RUTING/ROUTING: '{text}' -> {got} (ventet/expected {want})")
            bad += 1
    print(f"Rutingtest: {len(ROUTING_PROBES) - bad}/{len(ROUTING_PROBES)} riktige / routing check: {len(ROUTING_PROBES) - bad}/{len(ROUTING_PROBES)} correct")
    return bad


def code_check() -> int:
    """CodeAgent uten ekte Ollama: kanned svar, feilvei og ruting. / CodeAgent without real Ollama: canned answer, error path, routing."""
    import llm
    saved = (llm.installed_models, llm.chat, llm.hardware)
    bad = 0

    def check(name: str, cond: bool) -> None:
        nonlocal bad
        print(f"  {'OK ' if cond else 'FEIL/FAIL'} CodeAgent: {name}")
        bad += 0 if cond else 1

    try:
        llm.hardware = lambda: {"ram_gb": 32.0, "free_gb": 20.0, "vram_gb": 0.0, "gpu": ""}
        llm.installed_models = lambda: ["qwen2.5-coder:7b"]
        llm.chat = lambda msgs, model, *a, **k: "```ts\nexport const x = 1;\n```\nThis exports x."
        agent = CodeAgent()
        res = agent.run("Write a function that returns one")
        check("splits code and explanation", res["code"] == "export const x = 1;" and "exports x" in res["explanation"])
        check("remembers the last exchange", len(agent.history) == 2)
        agent._up = True
        r = Router(None)
        r.agents["code"]._up = True
        check("routes clear code requests", r.route("Write a clean Next.js API route that returns JSON").name == "code")
        check("does not steal music", r.route("write about my favourite blues records").name != "code")
        agent2 = CodeAgent()
        agent2._up = False
        check("stays out of routing when Ollama is down", agent2.relevance("Write a function that sorts") == 0.0)

        def down():
            raise llm.OllamaNotRunning(llm.NOT_RUNNING.format(h=llm.host()))
        llm.installed_models = down
        err = CodeAgent().run("Write a function")
        check("clear message when Ollama is down", "ollama.com/download" in err["explanation"] and not err["code"])
    finally:
        llm.installed_models, llm.chat, llm.hardware = saved
    return bad


def selftest() -> int:
    """Kjør alle oppgaver på norsk og engelsk og se etter typisk AI-språk, og sjekk rutingen.
    Style check for stock AI phrasing, plus a routing check (with models/libraries if they exist)."""
    router = Router(None)
    routing_bad = routing_check(build_router(quiet=True)) + routing_check(Router(None)) + code_check()
    probes = ["blues slide guitar ideas", "idéer til en bluesblogg om slide", "jazz listen recommend", "anbefal rock å lytte til",
              "I woke up this morning; I woke up this morning; blues walking like a man", "metadata for en jazzlåt",
              "seo checklist", "seo for nettsiden min", "stripe checkout", "stripe betaling", "next.js app router", "next.js på norsk",
              "should i use cloudflare or vercel", "hvilken hosting bør jeg velge cloudflare"]
    bad = 0
    for q in probes:
        r = router.run(q, agent="web" if any(k in q for k in ("seo", "stripe", "next", "hosting", "cloudflare")) else "music")
        blob = " ".join(l for _, ls in r["sections"] for l in ls).lower()
        blob += " " + " ".join(t[lang].lower() for g in GENRES.values() for key in ("listen", "hooks") for t in g[key] for lang in ("en", "no"))
        blob += " " + " ".join(g["intro"][lang].lower() for g in GENRES.values() for lang in ("en", "no"))
        blob += " " + " ".join(t[lang].lower() for _, t in CRAFT_TIPS for lang in ("en", "no"))
        blob += " " + " ".join(t[lang].lower() for t in WEB_NEXT.values() for lang in ("en", "no"))
        for b in BANNED:
            if b in blob:
                print(f"STIL/STYLE: '{b}' in output for: {q}")
                bad += 1
    print("Stiltest OK / style test OK" if not bad else f"{bad} treff / hits")
    return 1 if (bad or routing_bad) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="KM-TECH BRAIN agents (lokalt / local)")
    p.add_argument("text", nargs="*")
    p.add_argument("--agent", choices=sorted(REGISTRY), help="Velg agent / force an agent")
    p.add_argument("--task", help="Velg oppgave / force a task (ideas, lyrics, metadata, listen, seo, stripe, nextjs, decision)")
    p.add_argument("--model", default="models/brain.pth")
    p.add_argument("--library", default="data/music_genres.csv",
                   help="Eksempler MusicAgent søker i / examples MusicAgent searches (python retrieve.py)")
    p.add_argument("--web-library", default="data/web_topics.csv", help="Eksempler WebAgent søker i / examples WebAgent searches")
    p.add_argument("--genre-model", default="models/genres.pth", help="Valgfri sjangermodell / optional genre model")
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

    router = build_router(a.model, a.library, a.web_library, a.genre_model)

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
