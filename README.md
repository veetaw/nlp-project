# Analisi Temporale del Sentiment di post su Bluesky

Questo repository contiene una pipeline completa di Elaborazione del Linguaggio Naturale (NLP) per l'estrazione, il preprocessing, l'analisi del sentiment e la modellazione dei topic dei post scritti in lingua italiana sulla piattaforma decentralizzata **Bluesky**. Il caso di studio analizzato riguarda la figura della Presidente del Consiglio italiana **Giorgia Meloni** nell'intervallo temporale tra il 1° gennaio 2026 e il 18 giugno 2026.

## Struttura del Progetto

Il repository è organizzato come segue:

```text
├── datasets/                 # Dataset storici e temporali (file .csv)
├── output/                   # File di output dell'analisi e grafici generati
├── relazione/                # File sorgente LaTeX (relazione.tex) e PDF compilato
├── src/                      # Codice sorgente Python della pipeline
│   ├── config.py             # Configurazioni globali del progetto (chiavi, modelli, percorsi)
│   ├── data_extractor.py     # Estrazione dati tramite API di Bluesky (ricerca e time series)
│   ├── preprocessing.py      # Modulo per la pulizia del testo e normalizzazione
│   ├── models.py             # Classificatori di sentiment e funzioni di score continuo
│   ├── main.py               # Script principale per l'esecuzione della pipeline
│   ├── downloader.py         # Script helper per download di dataset alternativi
│   └── keywords_extract.py   # Script helper per l'estrazione semantica delle keyword
├── data.ipynb                # Notebook per l'analisi dei dati, spike detection e topic modeling
├── results.ipynb             # Notebook per la visualizzazione e generazione dei grafici
├── requirements.txt          # Dipendenze Python necessarie per eseguire il progetto
└── README.md                 # Questo file
```

---

## Architettura della Pipeline

La pipeline si sviluppa in cinque fasi principali:

1. **Selezione delle Keyword ed Estrazione**:
   * Partendo da un dataset pilota (1.000 post), viene applicata l'espansione semantica tramite **KeyBERT** (con modello `intfloat/multilingual-e5-small`) filtrando con le stopword italiane di NLTK.
   * Selezione delle parole chiave finali (`'meloni'`, `'governo meloni'`, `'giorgia meloni'`) per evitare rumore off-topic.
   * Download della serie temporale (13.465 post unici) tramite le API di ricerca di Bluesky.

2. **Preprocessing**:
   * Pulizia del testo: rimozione di URL e delle menzioni utente (`@username`).
   * Normalizzazione degli hashtag: rimozione del carattere `#` per preservare il valore semantico delle parole.
   * Conversione del testo in minuscolo.

3. **Sentiment Analysis e Score Continuo**:
   * Calcolo della probabilità delle tre classi (positivo, negativo, neutrale) confrontando un modello specifico italiano (`feel-it-italian-sentiment`) e uno multilingua (`twitter-roberta-base-sentiment-latest`).

4. **Validazione LLM-as-a-Judge**:
   * Per risolvere le discordanze di classificazione tra i due modelli (12.550 casi su 13.465), viene impiegato il modello locale quantizzato **Qwen 3.5-4B** (eseguito a temperatura `0.1`) per valutare un campione di 2.711 post.
   * Il modello multilingua è stato eletto come il più accurato (1.935 voti contro 776) ed è stato usato per il resto dell'analisi.

5. **Spike Detection e Topic Modeling**:
   * Identificazione automatica dei picchi di volume di post giornalieri per ciascuna classe di sentiment (valori superiori a 1.5 deviazioni standard sopra la media).
   * Estrazione dei temi caldi tramite **BERTopic** applicato olisticamente a tutti i post pubblicati nei giorni di spike (coadiuvato dalle stopword italiane di NLTK).
   * Analisi comparativa finale dei token più frequenti per ciascuna classe di sentiment.

---

## Modelli Utilizzati

* **Embedding e Keyword Extraction**: `intfloat/multilingual-e5-small` (tramite KeyBERT).
* **Sentiment Analysis (Italiano)**: `MilaNLProc/feel-it-italian-sentiment`.
* **Sentiment Analysis (Multilingua)**: `cardiffnlp/twitter-roberta-base-sentiment-latest`.
* **LLM**: `Qwen 3.5-4B` (locale).
* **Topic Modeling**: `BERTopic` (con modelli di embedding multilingua e riduzione dimensionale UMAP/HDBSCAN).

---

## Come Eseguire il Progetto

### 1. Configurazione dell'Ambiente

Assicurati di avere Python 3.10+ installato. Crea un ambiente virtuale e installa le dipendenze:

```bash
# Creazione ambiente virtuale
python -m venv .venv

# Attivazione ambiente virtuale (Windows)
.venv\Scripts\activate

# Installazione dipendenze
pip install -r requirements.txt
```

### 2. Configurazione dei Parametri e Credenziali

Apri il file `src/config.py` e configura le tue credenziali di accesso Bluesky e i parametri di esecuzione:

```python
BLUESKY_HANDLE = "tuo_handle.bsky.social"
BLUESKY_PASSWORD = "tua_app_password"
```

### 3. Esecuzione della Pipeline

Per eseguire l'estrazione, il preprocessing e la classificazione del sentiment tramite script:

```bash
python src/main.py
```

I risultati verranno salvati in `output/meloni_with_sentiment.csv`.

### 4. Esplorazione dei Notebook

Per riprodurre le analisi dettagliate sugli spike, su BERTopic e generare i grafici per la relazione, avvia Jupyter ed esplora i notebook nella cartella radice:

```bash
jupyter notebook
```
* **`data.ipynb`**: si occupa del rilevamento dei picchi temporali e del topic modeling.
* **`results.ipynb`**: genera i grafici di distribuzione e andamento temporale del sentiment.
