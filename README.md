# KM-TECH BRAIN — lokal AI-hjerne / local AI brain (PyTorch)

Fullt offline. Ingen API-er, ingen kreditter. / Fully offline. No APIs, no credits.

## Filer / Files

| Fil / File | NO | EN |
|---|---|---|
| `model.py` | Nettverk: `TextBrain` (embedding → mean pooling → MLP) og `DenseBrain` (egne vektorer → MLP). Dropout, BatchNorm, klassifisering + regresjon. | Networks: `TextBrain` (embedding → mean pooling → MLP) and `DenseBrain` (your own vectors → MLP). Dropout, BatchNorm, classification + regression. |
| `data.py` | Leser CSV/JSON, tokenisering, vokabular, train/val-split, `add_example()`. | Reads CSV/JSON, tokenizer, vocabulary, train/val split, `add_example()`. |
| `train.py` | Treningsløkke: AdamW, LR-scheduler, early stopping, logging, lagrer beste modell til `.pth`. | Training loop: AdamW, LR scheduler, early stopping, logging, saves best model to `.pth`. |
| `predict.py` | Laster `.pth`, gir prediksjon + sikkerhet (enkel og batch). Klassen `Brain` kan brukes fra Python. | Loads `.pth`, returns prediction + confidence (single and batch). `Brain` class is usable from Python. |
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
