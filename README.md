# KM-TECH BRAIN — lokal AI-hjerne / local AI brain (PyTorch)

Fullt offline. Ingen API-er, ingen kreditter. / Fully offline. No APIs, no credits.

## Filer / Files

| Fil / File | NO | EN |
|---|---|---|
| `model.py` | Nettverk: `TextBrain` (embedding → mean pooling → MLP) og `DenseBrain` (egne vektorer → MLP). Dropout, BatchNorm, klassifisering + regresjon. | Networks: `TextBrain` (embedding → mean pooling → MLP) and `DenseBrain` (your own vectors → MLP). Dropout, BatchNorm, classification + regression. |
| `data.py` | Leser CSV/JSON, tokenisering, vokabular, train/val-split, `add_example()`. | Reads CSV/JSON, tokenizer, vocabulary, train/val split, `add_example()`. |
| `train.py` | Treningsløkke: AdamW, LR-scheduler, early stopping, logging, lagrer beste modell til `.pth`. | Training loop: AdamW, LR scheduler, early stopping, logging, saves best model to `.pth`. |
| `predict.py` | Laster `.pth`, gir prediksjon + sikkerhet (enkel og batch). Klassen `Brain` kan brukes fra Python. | Loads `.pth`, returns prediction + confidence (single and batch). `Brain` class is usable from Python. |
| `agents.py` | Lokale agenter (`MusicAgent`, `WebAgent`) som bruker modellen + innebygd fagkunnskap. | Local agents (`MusicAgent`, `WebAgent`) using the model + built-in domain know-how. |
| `web/` | Statisk landingsside (HTML/CSS/JS) klar for Cloudflare Pages. | Static landing page (HTML/CSS/JS), ready for Cloudflare Pages. |
| `utils.py` | Seed, enhet, tospråklige meldinger. | Seed, device, bilingual messages. |
| `data/example_seed.csv` | 100 håndskrevne eksempler (5 kategorier). | 100 hand-written examples (5 categories). |
| `data/make_example_data.py` | Bygger `example_data.csv` (500 rader) = seed + malgenererte setninger. | Builds `example_data.csv` (500 rows) = seed + template-generated sentences. |
| `data/example_data.csv` | Treningsdata for demoen. | Demo training data. |

Kategorier / Categories: Blues/Roots, Code/Tech, American Culture, Business/Monetization, Health/Productivity.

## Installasjon / Install

```bash
pip install -r requirements.txt
```

## Bruk / Usage

```bash
python train.py                                   # tren på eksempeldata / train on example data
python predict.py                                 # demo / demo
python predict.py "slide guitar på en gammel juke joint"
python predict.py "tekst 1" "tekst 2"             # batch / batch
python predict.py --file tekster.txt              # én tekst per linje / one text per line
python predict.py --interactive
```

Egne data / Your own data (`text,label` i CSV, eller liste av `{"text","label"}` i JSON):

```bash
python data.py add "Ny tekst her" "Blues/Roots" --file data/example_data.csv
python train.py --data data/example_data.csv --epochs 300 --out models/brain.pth
```

Nyttige flagg / Useful flags: `--hidden 256 128`, `--embed-dim 128`, `--dropout 0.5`, `--no-batchnorm`,
`--lr 1e-2`, `--patience 40`, `--val-split 0.2`, `--task regression` (labels må være tall / must be numbers).

## Laste modellen senere / Loading the model later

```python
from predict import Brain
brain = Brain("models/brain.pth")
brain.predict("Muddy Waters live in Chicago")
# {'label': 'Blues/Roots', 'confidence': 0.99, 'probabilities': {...}}
brain.predict_batch(["text a", "text b"])
```

Filen inneholder vekter, config, vokabular og klassenavn, så alt trengs kun `.pth`. /
The file holds weights, config, vocabulary and class names, so the `.pth` alone is enough.

## Utvide / Extending

* **Egne embeddings / own embeddings:** `TextBrain(..., pretrained=tensor[vocab, dim])`, eller bruk `DenseBrain` på ferdige vektorer. / or use `DenseBrain` on precomputed vectors.
* **Ærlig merknad / Honest note:** `example_data.csv` er delvis syntetisk (maler), så valideringsscore (~95 %+) er optimistisk. Med bare de 100 håndskrevne eksemplene ligger den rundt 30–60 %. Legg til flere ekte eksempler. /
  `example_data.csv` is partly synthetic (templates), so its validation score (~95%+) is optimistic. With only the 100 hand-written examples it is around 30–60%. Add more real examples.

## Landingsside / Landing page (`web/`)

Ren HTML + CSS + litt JavaScript, ingen build, ingen backend. / Plain HTML + CSS + a little JS, no build, no backend.

```bash
python -m http.server 8080 --directory web
```
Åpne / open http://localhost:8080 (du kan også dobbeltklikke `web/index.html` / or just double-click `web/index.html`).

### Deploy til Cloudflare Pages / Deploy to Cloudflare Pages
1. Push repoet til GitHub. / Push the repo to GitHub.
2. Cloudflare dashboard → Workers & Pages → Create → Pages → Connect to Git → velg repoet / pick the repo.
3. Framework preset: **None**. Build command: *(tom / empty)*. Build output directory: **`web`**.
4. Deploy. (Eller / or: `npx wrangler pages deploy web`.) `web/_headers` setter sikkerhets- og cache-headere / sets security and cache headers.

## Agenter / Agents

En agent = nøkkelord + oppgaver ("tasks") + valgfri bruk av den trente modellen (`models/brain.pth`).
Agentene er regelbaserte og lokale (ingen API-er) og genererer ikke fri tekst: modellen klassifiserer, agenten gir sjekklister, analyser og ideer.
An agent = keywords + tasks + optional use of the trained model. Agents are rule-based and local (no APIs) and do not generate free text: the model classifies, the agent returns checklists, analyses and ideas.

```bash
python agents.py --list
python agents.py "ideas for a blues blog about Delta slide guitar"      # auto-ruting / auto-routing
python agents.py "How do I set up Stripe checkout on Cloudflare?"
python agents.py --agent music --task lyrics "I woke up this morning; I woke up this morning; Blues walking like a man"
python agents.py --agent music --task metadata "Slow Delta blues about a long road home."
python agents.py --agent web --task seo "SEO for my Next.js shop"
python agents.py --interactive
```

| Agent | Oppgaver / Tasks |
|---|---|
| `music` | `ideas`, `lyrics` (linjer, repetisjon, rim, AAB-struktur), `metadata` (tittel, tags, beskrivelse) |
| `web` | `seo`, `stripe`, `nextjs`, `decision` |

Ruting: modellens kategori (f.eks. Blues/Roots, Code/Tech) + nøkkelord velger agent. Uten treningsfil (`python train.py` ikke kjørt) bruker agentene kun nøkkelord. /
Routing: the model's category plus keywords pick the agent. Without a trained model the agents use keywords only.

Fra Python / From Python:
```python
from agents import Router, load_brain
router = Router(load_brain("models/brain.pth"))
print(router.run("ideas for a blues blog", agent="music"))
```

Ny agent / New agent: subclass `Agent` i `agents.py`, sett `name`, `keywords`, `domains`, `task_triggers`, legg til `task_<navn>()`-metoder og kall `register(MinAgent)`. /
subclass `Agent`, set the attributes, add `task_<name>()` methods, call `register(MyAgent)`.
