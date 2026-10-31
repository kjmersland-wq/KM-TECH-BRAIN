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
| `embeddings.py` | Encodere: `hashed` (lettvekts) og `pretrained` (sentence-transformers), med fallback. | Encoders: `hashed` (lightweight) and `pretrained` (sentence-transformers), with fallback. |
| `retrieve.py` | Finn lignende eksempler i biblioteket ditt (embeddings + cosinus). | Find similar examples in your library (embeddings + cosine). |
| `data/music_genres.csv` | 84 håndskrevne musikkeksempler: Blues, Jazz, Rock, Other (norsk + engelsk). | 84 hand-written music examples: Blues, Jazz, Rock, Other (Norwegian + English). |
| `data/eval_music.csv` | 24 testtekster for sjanger, ikke i treningsdata. | 24 genre test texts, not in the training data. |
| `evaluate.py` | Nøyaktighet per kategori, F1 og forvirringsmatrise. | Per-category accuracy, F1 and confusion matrix. |
| `data/eval_real.csv` | 50 håndskrevne testtekster (ikke i treningsdata). | 50 hand-written test texts (not in the training data). |
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

## Tekstforståelse / Text understanding (`--mode`)

| Modus / Mode | Hva / What | Ekte-tekst-test* / real-text test* |
|---|---|---|
| `simple` (standard) | Egen embedding trent fra scratch. Ingen ekstra pakker. / Own embedding, no extra packages. | 58 % |
| `hashed` | Lettvekts: hashede ord + tegn-n-gram. Kun torch, fungerer godt på norske ordformer. / Lightweight, torch only, good on Norwegian word forms. | **78 %** |
| `pretrained` | Ferdigtrente setnings-embeddings (`sentence-transformers`, flerspråklig inkl. norsk). / Pre-trained multilingual sentence embeddings. | ikke målt her / not measured here |

\* 50 håndskrevne norske/engelske tekster i `data/eval_real.csv` som modellen aldri har sett. Tallene er små og støyete. /
50 hand-written Norwegian/English texts in `data/eval_real.csv` the model never saw. Small, noisy numbers.

```bash
python train.py                                   # simple (som før / as before)
python train.py --mode hashed                     # anbefalt uten ekstra pakker / recommended, no extra packages
pip install -r requirements-pretrained.txt
python train.py --mode pretrained --download      # --download KUN første gang (laster ned ~120 MB modell) / ONLY the first time
python train.py --mode pretrained                 # videre helt offline / fully offline afterwards
```
`pretrained` uten pakken installert gir en tydelig melding og faller tilbake til `simple`. En modell trent med `hashed`/`pretrained` husker selv hvilken encoder den trenger, så `predict.py`/`agents.py` fungerer uendret. Du kan også peke `--encoder-model` på en lokal mappe. /
Without the package, `pretrained` prints a notice and falls back to `simple`. A trained model remembers its encoder, so `predict.py`/`agents.py` work unchanged. `--encoder-model` also accepts a local folder.

## Musikk-bibliotek og søk / Music library and similarity search

`data/music_genres.csv` er biblioteket MusicAgent bruker (Blues, Jazz, Rock, Other). Legg til dine egne ekte eksempler: titler, beskrivelser, tekstlinjer, artikler. /
`data/music_genres.csv` is the library MusicAgent uses. Add your own real examples: titles, descriptions, lyric snippets, articles.

```bash
python retrieve.py "slide guitar i åpen G på en gammel resonator"      # mest like eksempler / most similar
python retrieve.py "Cmaj7 voicings like Bill Evans" -k 3 --label Jazz
python retrieve.py --eval data/eval_music.csv                           # kNN-treff på testsettet / kNN accuracy
python retrieve.py "riff i drop D" --mode pretrained                    # auto | hashed | pretrained
```
`--mode auto` bruker `pretrained` hvis sentence-transformers er installert og modellen er lastet ned, ellers `hashed`. Indeksen caches i `models/`. Python: `Retriever(...).similar(text, k)` og `.vote(text)`. /
`auto` uses `pretrained` if installed and already downloaded, otherwise `hashed`. The index is cached in `models/`.

Sjangermodell (valgfri, brukes av MusicAgent hvis den finnes) / optional genre model, used by MusicAgent if present:
```bash
python train.py --data data/music_genres.csv --mode hashed --out models/genres.pth --epochs 400 --patience 60
python evaluate.py --model models/genres.pth --data data/eval_music.csv
```

## Mange eksempler på én gang / Adding many examples

```bash
python data.py import notes.txt --label Jazz --file data/music_genres.csv      # én tekst per linje / one text per line
python data.py import samples.txt --file data/music_genres.csv                  # linjer som: label<TAB>tekst  eller  label | tekst
python data.py import music_inbox/ --file data/music_genres.csv                 # undermappe = label: music_inbox/Blues/*.txt
python data.py import artikkel.txt --label Rock --unit chunks --chunk-words 60 --file data/music_genres.csv   # lange artikler i biter
python data.py import post.txt --label Blues --unit paragraphs --file data/music_genres.csv   # blank linje skiller eksempler
python data.py import music_inbox/ --dry-run --file data/music_genres.csv       # forhåndsvis / preview only
python data.py stats --file data/music_genres.csv                               # antall per sjanger / counts per genre
```
`--unit`: `lines` (standard for .txt), `paragraphs`, `file` (standard i undermapper), `chunks` (artikler delt ved setningsgrenser). Duplikater hoppes over. Uten `--file` brukes `data/example_data.csv`. /
Duplicates are skipped. Without `--file` the default is `data/example_data.csv`.

Hva som teller som et godt eksempel / what makes a good example: ekte tekst slik den faktisk dukker opp (plateomtaler, spørsmål på forum, tekstlinjer, artikkelavsnitt), 10–60 ord, og gjerne en blanding av norsk og engelsk. Over 30 eksempler per sjanger gjør en merkbar forskjell. /
Real text as it actually appears, 10–60 words, ideally mixed Norwegian and English. 30+ examples per genre makes a noticeable difference.

## Evaluering / Evaluation

```bash
python evaluate.py --data data/eval_real.csv --show-errors   # ANBEFALT / recommended: tekst modellen ikke har sett
python evaluate.py                                           # val-splitten fra treningen / training val split
```
Viser nøyaktighet, precision/recall/F1 per kategori og forvirringsmatrise. Python: `from evaluate import evaluate`. /
Shows accuracy, per-category precision/recall/F1 and a confusion matrix.
Tips: legg dine egne ekte tekster i en egen fil som du aldri trener på, og evaluer mot den. / Keep a file of your own real texts that you never train on, and evaluate against it.

## Agenter / Agents

En agent = nøkkelord + oppgaver ("tasks") + forslag per kategori + valgfri bruk av den trente modellen.
Agentene er lokale og regelbaserte (ingen API-er). Modellen gir kategori og sikkerhet; agenten gir **håndskrevne** forslag og neste steg.
An agent = keywords + tasks + suggestions per category + optional use of the trained model. Local and rule-based. The model gives category and confidence; the agent gives **hand-written** suggestions and next steps.

* **Norsk og engelsk:** språket oppdages automatisk og svaret kommer på samme språk. / Language is detected and the answer comes back in the same language.
* **Sikkerhet:** kategori og prosent vises; under 50 % sier agenten ifra og ber om litt mer informasjon. / Category and confidence are shown; under 50% the agent says so and asks for more.
* **Minne:** i `--interactive` (og via `Router`) husker agenten de 5 siste meldingene. Korte oppfølgere som «mer», «hva med SEO?» arver tema, agent og språk. `/historikk` viser minnet, `/glem` tømmer det. / In `--interactive` (and via `Router`) the agent remembers the last 5 messages; short follow-ups inherit topic, agent and language.
* **Skrivestil:** musikktekster (blues, jazz, rock) er skrevet som en entusiast med konkrete plater og detaljer, ikke som en chatbot, og norsk er skrevet på norsk, ikke oversatt. `python agents.py --selftest` ser etter typiske AI-fraser (`BANNED` i `agents.py`). / Music text is written like an enthusiast, with specific records and details; Norwegian is written as Norwegian, not translated. `--selftest` scans for stock AI phrases.

```bash
python agents.py --list
python agents.py "ideer til en bluesblogg om slide-gitar"                 # auto-ruting, svar på norsk
python agents.py "recommend some jazz to listen to"
python agents.py --agent music --task lyrics "I woke up this morning; I woke up this morning; Blues walking like a man"
python agents.py --agent music --task metadata "Slow Delta blues about a long road home."
python agents.py "SEO for nettbutikken min i Next.js"
python agents.py "Hva er jazz egentlig? Forklar for en nybegynner"   # explain + sjanger + lignende fra biblioteket
python agents.py "finn lignende eksempler: slide i åpen D"
python agents.py --interactive                                             # med minne / with memory
python agents.py --model models/brain.pth --selftest
```

| Agent | Oppgaver / Tasks |
|---|---|
| `music` | `ideas`, `listen` (lyttetips for blues/jazz/rock), `explain` (kort introduksjon til sjangeren), `related` (lignende eksempler fra biblioteket), `lyrics` (linjer, repetisjon, rim, AAB), `metadata` (tittel, tagger, beskrivelse) |
| `web` | `seo`, `stripe`, `nextjs`, `decision` |

**MusicAgent og sjanger / and genre:** sjangeren (blues, jazz, rock) er hovedsignalet. Den finnes i denne rekkefølgen: (1) nøkkelord og artistnavn, (2) sjangermodellen `models/genres.pth` hvis den finnes, (3) avstemning blant de mest like eksemplene i biblioteket (`--library`). Agenten sier hvordan den kom fram til sjangeren, og viser de mest like eksemplene dine under svaret. Uten bibliotek fungerer den på nøkkelord alene. /
Genre is the main signal, found by (1) keywords and artist names, (2) the optional genre model, (3) a vote among the most similar library examples. The agent tells you how it decided and shows the closest examples from your library. Without a library it works on keywords alone.

WebAgent er uendret. / WebAgent is unchanged.

Uten trent modell bruker agentene bare nøkkelord. / Without a trained model the agents use keywords only.

```python
from agents import Router, load_brain
router = Router(load_brain("models/brain.pth"))
res = router.run("ideas for a jazz blog")        # res["classification"], res["sections"], res["next"]
res = router.run("more")                         # oppfølger / follow-up, bruker minnet
```

Ny agent / New agent: subclass `Agent` i `agents.py`, sett `name`, `keywords`, `domains`, `task_triggers`, `suggestions`, legg til `task_<navn>(self, text, c)`-metoder og kall `register(MinAgent)`. /
Subclass `Agent`, set the attributes, add `task_<name>(self, text, c)` methods, call `register(MyAgent)`.
