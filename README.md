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
| `ask.py` | Enkleste inngang: sjanger/tema + lignende eksempler + kort tanke. | Easiest entry point: genre/topic + similar examples + a short take. |
| `ingest.py` | Slipp filer i `inbox/` og importer med oppsummering. | Drop files in `inbox/` and import with a summary. |
| `llm.py` | Klient for lokal LLM via Ollama (modellvalg, anbefaling, test). | Client for a local LLM via Ollama (model choice, recommendation, test). |
| `data/web_topics.csv` | 60 håndskrevne web-eksempler: SEO, Frontend, Backend, Payments, Hosting. | 60 hand-written web examples. |
| `data/eval_web.csv` | 15 testtekster for web, ikke i treningsdata. | 15 web test texts, not in the training data. |
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


## Daglig bruk / Everyday use

Det enkleste: **`python ask.py`**. Skriv et spørsmål eller en tekst på norsk eller engelsk, og få sjanger eller tema, lignende eksempler fra dine egne data og en kort tanke. /
The easiest way: **`python ask.py`**. Type a question or text in Norwegian or English and get the genre or topic, similar examples from your own data, and a short take.

```bash
python ask.py "slide i åpen D på en gammel resonator"      # ett spørsmål / one question
python ask.py                                               # samtale med minne / conversation with memory
python ask.py --full "ideas for a jazz blog"                # hele agentsvaret / the agent's full answer
```
I samtalen: `mer` / `more` gir hele svaret, `/historikk` viser minnet, `/glem` tømmer det, tom linje avslutter. Oppfølgere som «hva med SEO for den?» husker temaet. /
In the conversation: `more` gives the full answer, `/history` shows memory, `/reset` clears it, an empty line quits.

## Legg inn dine egne tekster / Adding your own real texts (the easy way)

1. Lag mappene én gang / create the folders once: `python ingest.py --init`
2. Slipp filene (`.txt`, `.md`, `.csv`, `.json`) i riktig undermappe / drop files in the right subfolder. Mappenavnet er kategorien / the folder name is the category:
   ```
   inbox/music/Blues/    inbox/music/Jazz/    inbox/music/Rock/    inbox/music/Other/
   inbox/web/SEO/   inbox/web/Frontend/   inbox/web/Backend/   inbox/web/Payments/   inbox/web/Hosting/
   ```
3. Kjør / run: `python ingest.py` (eller / or `python data.py inbox`). Vil du se først / preview first: `--dry-run`.

Du får en oppsummering per bibliotek og kategori: lagt til, duplikater, for korte, og totalen. Små skrivefeil i mappenavn rettes eller flagges («Jaz» → mente du «Jazz»?). Ferdige filer flyttes til `inbox/_imported/` (`--keep` for å la dem ligge). Nye kategorier krever `--allow-new`. /
You get a summary per library and category: added, duplicates, too short, and the total. Typos in folder names are caught ("Jaz" → did you mean "Jazz"?). Done files move to `inbox/_imported/`.

Hvordan en fil deles opp / how a file is split:

| Filen ser slik ut / The file looks like | Resultat / Result |
|---|---|
| `navn.lines.txt` | én tekst per linje / one example per line |
| blanke linjer mellom avsnitt / blank lines between paragraphs | ett eksempel per avsnitt (bra for tekststrofer / good for lyric stanzas) |
| kort fil, ca. 120 ord eller mindre / short file | hele filen er ett eksempel / the whole file is one example |
| lang fil / long file | biter på ca. 60 ord ved setningsgrenser / ~60-word chunks |
| direkte i `inbox/music/` med linjer `Jazz<TAB>tekst` eller `Jazz \| tekst` | kategorien står på linjen / the category is on the line |

`python ingest.py --retrain` trener sjangermodellen på nytt etterpå. Søk og agenter bruker de nye eksemplene med en gang uten trening. /
`--retrain` retrains the genre model afterwards. Search and agents use new examples immediately, without training.

Bibliotekene / the libraries: `data/music_genres.csv` (Blues, Jazz, Rock, Other) og / and `data/web_topics.csv` (SEO, Frontend, Backend, Payments, Hosting). Testsett som ikke brukes til trening / held-out test sets: `data/eval_music.csv`, `data/eval_web.csv`. kNN-treff / kNN accuracy (`python retrieve.py --data ... --eval ...`): musikk 71 % (24 tekster), web 53 % (15 tekster). Små, støyende tall; web er svakest og trenger flere ekte eksempler / music 71% (24 texts), web 53% (15 texts). Small, noisy numbers; web is weakest and needs more real examples.

Ruting mellom agentene / routing: `python agents.py --selftest` kjører 24 rutingtester (norsk og engelsk, med og uten modeller) pluss stilsjekken. Tekniske ord som SEO, Stripe og deploy trumfer sjangerord, så «SEO for bluesbloggen min» går til WebAgent og «slide i åpen G» til MusicAgent. De 24 testsetningene er de jeg justerte mot, så de er en regresjonssjekk, ikke et uavhengig mål. /
24 routing probes (Norwegian and English, with and without models) plus the style check. Technical terms beat genre words. Those probes are the ones I tuned against, so they are a regression check, not an independent measure.

## Lokal kode-LLM / Local code LLM (Ollama)

`CodeAgent` bruker en modell som kjører lokalt i [Ollama](https://ollama.com). Alt blir på maskinen din, ingen skytjenester, ingen kreditter. /
`CodeAgent` uses a model running locally in Ollama. Everything stays on your machine.

**1. Installer Ollama / Install Ollama.** Last ned fra https://ollama.com/download (Windows: kjør installasjonsprogrammet; Ollama starter i bakgrunnen og lytter på `http://localhost:11434`). Alternativt i terminal: `winget install Ollama.Ollama`. /
Download it from https://ollama.com/download; it then runs in the background on `http://localhost:11434`.

**2. Se hva som passer din maskin / see what fits your machine:**
```bash
python llm.py --models
```
Viser RAM og GPU, hva som er installert, og sterkeste modell som passer. Rekkefølgen jeg prøver / the order it tries: `qwen2.5-coder:14b` → `qwen2.5-coder:7b` → `deepseek-coder-v2` → `llama3.1`, og to små reserver / plus two small fallbacks (`qwen2.5-coder:3b`, `:1.5b`) for svake maskiner / for weak machines.

**3. Last ned en modell / pull a model** (kun første gang, stor nedlasting / one-time, large download):
```bash
ollama pull qwen2.5-coder:7b       # ca. 4,7 GB. Bruk :14b (ca. 9 GB) hvis du har 16 GB RAM eller 10+ GB VRAM
ollama pull qwen2.5-coder:3b       # ca. 1,9 GB, raskere og lettere, men svakere / faster and lighter, but weaker
```

**4. Test / test:**
```bash
python llm.py "Write a clean Next.js API route that returns JSON"
python llm.py --model qwen2.5-coder:3b "Write a debounce function in TypeScript"
python llm.py --selftest            # tester klienten mot en lokal stub, uten Ollama / tests the client against a local stub
```
Modellvalg / model choice: `--model`, ellers miljøvariabelen `KMBRAIN_MODEL`, ellers automatisk. Annen adresse: `OLLAMA_HOST`. Kjører ikke Ollama, får du en klar melding med hva du skal gjøre. /
Model: `--model`, else `KMBRAIN_MODEL`, else automatic. Other address: `OLLAMA_HOST`. If Ollama isn't running you get a clear message.

**5. CodeAgent:**
```bash
python agents.py --agent code "Write a clean Next.js API route that returns JSON"
python ask.py --agent code "Skriv en funksjon som validerer et norsk telefonnummer"
python ask.py --agent code          # samtale: «gjør den om til TypeScript» bygger på forrige svar / follow-ups build on the last answer
```
Du får kode og en kort forklaring (på språket du skriver). **Ingen filer endres.** Tydelige kodebestillinger («write a function…», «skriv en funksjon…») rutes automatisk til CodeAgent når Ollama kjører; ellers bruk `--agent code`. /
You get code plus a short explanation. **No files are changed.** Clear code requests are routed to CodeAgent automatically when Ollama is running.

### Realistiske forventninger / Realistic expectations

* **Bra nok til mye:** små funksjoner, regex, SQL, enkle API-ruter, boilerplate, omskriving mellom språk, forklaring av kode. Alt kjører lokalt og gratis. /
  **Good enough for plenty:** small functions, regex, SQL, simple API routes, boilerplate, translating between languages, explaining code.
* **Ikke på Claude-nivå:** de lokale modellene er mye mindre. Forvent svakere resonnering på flerfils-design og subtile feil, utdaterte eller oppdiktede API-er (spesielt nye versjoner av rammeverk som Next.js), og kortere effektiv kontekst. `qwen2.5-coder:14b` er merkbart bedre enn `7b`, og begge ligger et godt stykke under Claude. Les og kjør alltid koden selv. /
  **Not Claude-level:** these models are far smaller. Expect weaker reasoning on multi-file design and subtle bugs, outdated or invented APIs (especially recent framework versions), and shorter useful context. Always read and run the code yourself.
* **Hastighet:** avhenger av maskinen. Uten stor GPU kjører modellen på CPU, og et svar på 40 linjer kan ta flere minutter med 7B. Første spørsmål er tregest fordi modellen lastes inn. /
  **Speed** depends on the machine; on CPU a 40-line answer can take minutes with a 7B model, and the first question is slowest because the model loads.
* **På denne maskinen / on this machine** (7,9 GB RAM, GTX 1050 med 2 GB VRAM, i3-10100F): `14b` og `deepseek-coder-v2` er for store. `qwen2.5-coder:7b` er den sterkeste som i det hele tatt går, men det blir trangt og tregt (CPU, lite ledig RAM). `qwen2.5-coder:3b` er det trygge valget. Dette er en vurdering ut fra størrelser og maskinvare, ikke en måling med en ekte modell. /
  `14b` and `deepseek-coder-v2` are too big. `7b` is the strongest that runs at all, tight and slow; `3b` is the safe choice. This is an estimate from sizes and hardware, not a measurement with a real model.
