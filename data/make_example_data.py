"""Lager data/example_data.csv = håndskrevne eksempler (example_seed.csv) + malbaserte setninger.
Builds the example dataset = hand-written seed + template-generated sentences (synthetic, for a
realistic demo size). Replace with your own real data for real use.

Kjør / Run:  python data/make_example_data.py
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
rng = random.Random(7)

# label -> (templates, {slot: options})
SPEC = {
    "Blues/Roots": (
        ["{artist} and the sound of {style}", "Learn {style} on {instr} with {tech}",
         "A night of {style} at a {venue}", "The story of {artist} and early {style}",
         "{tech} lessons for {style} {instr} players", "Why {style} still matters: {artist} and the {venue}",
         "Hvordan spille {style} på {instr} med {tech}", "{artist} live: {style} fra {venue}"],
        {"artist": ["Robert Johnson", "Muddy Waters", "B.B. King", "Howlin' Wolf", "Lead Belly", "Son House",
                    "Buddy Guy", "Albert King", "John Lee Hooker", "Skip James", "Etta James", "Bessie Smith"],
         "style": ["Delta blues", "Chicago blues", "Piedmont blues", "country blues", "electric blues",
                   "Texas blues", "roots music", "gospel blues", "slide guitar blues", "boogie woogie"],
         "instr": ["acoustic guitar", "harmonica", "resonator guitar", "piano", "slide guitar", "bass"],
         "tech": ["twelve bar progressions", "bending notes", "fingerpicking", "open tunings", "shuffle rhythm",
                  "call and response", "the pentatonic scale", "turnarounds"],
         "venue": ["juke joint", "Mississippi crossroads", "Memphis club", "front porch", "blues festival",
                   "Chicago bar", "bluesfestival"]}),
    "Code/Tech": (
        ["How to {task} in {lang}", "{lang} tutorial: {task} with {tool}", "Fixing {bug} when you {task}",
         "Deploy your {lang} app using {tool}", "{tool} vs {tool2} for {task}", "Best practices to {task} in {lang}",
         "Hvordan {task} med {lang} og {tool}", "Debugging {bug} in a {lang} project"],
        {"task": ["build an API", "train a neural network", "write unit tests", "parse JSON", "optimize queries",
                  "handle async requests", "containerize a service", "automate deployment", "manage state",
                  "refactor legacy code"],
         "lang": ["Python", "TypeScript", "Rust", "Go", "JavaScript", "Java", "C++", "SQL"],
         "tool": ["Docker", "PyTorch", "React", "Git", "Kubernetes", "Postgres", "FastAPI", "Linux", "Node.js"],
         "tool2": ["Redis", "Webpack", "Terraform", "GitHub Actions", "Vite", "Nginx"],
         "bug": ["a null pointer error", "a memory leak", "a race condition", "a failing build",
                 "a segmentation fault", "a type error", "a slow query"]}),
    "American Culture": (
        ["{event} in {place}", "A guide to {event} and {food}", "Celebrating {event} with {food} in {place}",
         "{place}: {food}, {sport} and local traditions", "Why Americans love {sport} and {event}",
         "En tur til {place} med {food} og {sport}", "The history of {event} across {place}"],
        {"event": ["Thanksgiving", "the Fourth of July", "Halloween", "Super Bowl Sunday", "Memorial Day",
                   "Mardi Gras", "a county fair", "the rodeo", "Labor Day", "Independence Day parades"],
         "place": ["Texas", "New Orleans", "Nashville", "New York", "the Midwest", "Route 66", "Memphis",
                   "Hollywood", "the Deep South", "Yellowstone"],
         "food": ["barbecue", "pumpkin pie", "hot dogs", "biscuits and gravy", "cornbread", "apple pie",
                  "fried chicken", "diner coffee"],
         "sport": ["baseball", "football", "basketball", "NASCAR", "rodeo", "high school football"]}),
    "Business/Monetization": (
        ["How to increase {metric} with {channel}", "{channel} strategy to boost {metric}",
         "Building a {model} business: pricing and {metric}", "Monetize your {asset} using {channel}",
         "Case study: doubling {metric} through {channel}", "Tjene penger på {asset} med {channel}",
         "Why {metric} matters for a {model} startup"],
        {"metric": ["revenue", "conversion rate", "profit margin", "customer lifetime value", "sales",
                    "monthly recurring revenue", "ROI", "omsetning"],
         "channel": ["email marketing", "SEO", "affiliate links", "paid ads", "sponsorships", "a sales funnel",
                     "AdSense", "Patreon", "Stripe checkout"],
         "model": ["SaaS", "subscription", "ecommerce", "freelance", "membership", "marketplace"],
         "asset": ["blog", "newsletter", "online course", "app", "podcast", "music catalog", "nettside"]}),
    "Health/Productivity": (
        ["{habit} to improve {goal}", "How {habit} helps {goal}", "A simple {habit} routine for better {goal}",
         "Science of {habit}: effects on {goal}", "{habit} and {habit2} for {goal}",
         "Slik forbedrer {habit} din {goal}", "Beat stress with {habit} and protect your {goal}"],
        {"habit": ["daily walking", "meditation", "strength training", "time blocking", "a sleep schedule",
                   "yoga", "journaling", "intermittent fasting", "deep work sessions", "stretching", "trening"],
         "habit2": ["a balanced diet", "breathing exercises", "regular breaks", "hydration", "a weekly review"],
         "goal": ["focus", "sleep quality", "mental health", "energy levels", "productivity", "back pain",
                  "mood", "long term health", "concentration", "søvn"]}),
}


def fill(template, slots):
    out = template
    for k, opts in slots.items():
        if "{" + k + "}" in out:
            out = out.replace("{" + k + "}", rng.choice(opts), 1)
    return out


rows = list(csv.DictReader((HERE / "example_seed.csv").open(encoding="utf-8")))
for label, (templates, slots) in SPEC.items():
    seen = set()
    while len(seen) < 80:
        seen.add(fill(rng.choice(templates), slots))
    rows += [{"text": t, "label": label} for t in sorted(seen)]
rng.shuffle(rows)
with (HERE / "example_data.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["text", "label"])
    w.writeheader()
    w.writerows(rows)
print(f"Skrev/Wrote {len(rows)} rows -> data/example_data.csv")
