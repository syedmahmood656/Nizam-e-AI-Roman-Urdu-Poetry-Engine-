# Nizam-e-AI-Roman-Urdu-Poetry-Engine-
An End-to-End NLP Pipeline for Author Identification &amp; Semantic Recommendation This project tackles the unique challenges of Roman-Urdu and provides two core functionalities:Multiclass Classification: Predicting the poet(Author) Semantic Recommendation: A vector-based retrieval system that recommends similar ash'aar(verses) when given a single word

# PROJECT OVERVIEW

Roman-Urdu Poetry Analysis, Author Identification & Recommendation
# Project Summary:

Nizam-e-AI is a hybrid Roman-Urdu poetry system that combines exact keyword search, TF‑IDF baselines,
and Sentence‑BERT semantic embeddings to provide author identification, poem recommendations, and an
interactive Streamlit demo.

# Features:

Author Identification — multiclass classifier (TF‑IDF + LogisticRegression baseline; SBERT
embeddings + LR for a stronger pipeline).

Semantic Recommendation — Sentence‑BERT embeddings + nearest‑neighbor retrieval for
meaning-based poem similarity.

Exact Keyword Search — word-boundary, case-insensitive matching for precise lookup.

Hybrid Search — keyword-first, semantic-fallback retrieval strategy.

Interactive UI — Streamlit app for real-time inference and exploration.
# Why this project matters:

Solves a real low-resource NLP problem (Roman‑Urdu).

Demonstrates full ML lifecycle: data collection, cleaning, feature engineering, modeling, evaluation,
explainability, and deployment.

Uses both classic NLP and modern transformer-based techniques — shows practical model
comparison and trade-offs.

Clear, deployable demo that hiring managers can run in minutes.

# Quick results (example highlights you can paste into your resume)

Baseline TF‑IDF + Logistic Regression: macro-F1 = X.XXX (example placeholder).

SBERT embeddings + LR classifier: macro-F1 = Y.YYY (improves over baseline; include your actual
numbers and dataset size).

Hybrid retrieval provides precise keyword matches and improved semantic relevance in top‑5
recommendations.

# Getting started (developer instructions)
Prerequisites
Python 3.8+
Colab or local machine (CPU okay; GPU recommended for faster embedding generation)
Git, GitHub account for deployment (Streamlit Cloud / Hugging Face Spaces) 

##  Installation

```bash
pip install -r requirements.txt
```

## requirements.txt should include:  
streamlit
pandas
numpy
scikit-learn
joblib
sentence-transformers
torch

## File layout (repo root)


```
.
app.py
poetry_data.csv # cleaned roman-Urdu dataset
nizam_e_ai_author_model.pkl # TF-IDF + LR baseline
nizam_e_ai_tfidf.pkl # TF-IDF vectorizer
nizam_sbert_author_clf.pkl # SBERT embedding + LR classifier
nizam_author_encoder.pkl # Label encoder joblib
nizam_poem_embeddings.npy # precomputed SBERT embeddings
requirements.txt
README.md
```

## How to run locally (development)

1. **Create virtualenv and install requirements.**
2. **Place poetry_data.csv and model files in repo root (or run training notebook to create them).**
3. **Run the Streamlit app:**
4. **streamlit run app.py**
5. **Open 'http://localhost:8501' in browser.**


## How to run/train from scratch (notebook)

A Colab notebook is included to load raw Kaggle data, perform normalization, train TF‑IDF baseline,
generate SBERT embeddings, train classifier on embeddings, and save artifacts. Follow cells in order:
Data → EDA → Baseline → SBERT embeddings → Embedding classifier → Save artifacts.

## Design & implementation details (short)

Preprocessing: lowercase, remove URLs/digits, collapse repeated characters, remove extraneous
punctuation while preserving line breaks, optional transliteration handling.
Baseline: TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
max_features=50000) + LogisticRegression(class_weight='balanced') — robust for
noisy roman text.
Semantic path: paraphrase-multilingual-MiniLM-L12-v2 (Sentence‑BERT) for embeddings.
Precompute and persist embeddings to .npy .
Author classifier (embedding): logistic regression on SBERT embeddings (fast, reproducible;
optional: LightGBM/SVM).
Recommendation: cosine similarity on SBERT embeddings + NearestNeighbors for scaling.
Keyword search: regex with word-boundaries, optional normalization toggle for precision. 

## Deployment notes

Streamlit Community Cloud: easiest for demos. Push repo to GitHub, point Streamlit Cloud to
app.py .
Hugging Face Spaces: alternative (supports Streamlit).
For production: containerize ( Dockerfile ), use a small API backed by fast vector DB (FAISS /
Milvus) for scaling, and add caching for embeddings and results.

## Resume bullets (copy-paste)

Built Nizam-e-AI, a Roman‑Urdu poetry retrieval system combining keyword search and
transformer-based semantic retrieval, deployed as a Streamlit app.
Implemented author identification (TF‑IDF baseline; Sentence‑BERT embeddings), achieving
improved macro‑F1 over baseline.
Designed hybrid retrieval that prioritizes exact keyword precision and falls back to semantic
similarity for recall.

## Contact & contribution

Add your GitHub link and email here.
If you accept contributions: add a CONTRIBUTING.md and a short code of conduct.

## Architecture diagram (ASCII + explanation)
```bash
+---------------------+ +----------------------+
+--------------------+
| Data Sources | ---> | Preprocessing | ---> | Persisted
Dataset |
| (Kaggle / Rekhta) | | - normalize roman | |
(poetry_data.csv) |
+---------------------+ | - dedupe / split |
+--------------------+
 |
 v
 +---------------------------+ <-- saved artifacts
 | Two Model Pipelines | - TF-IDF + LR (baseline)
 | 1) TF-IDF -> LR | - SBERT embeddings (.npy)
```

```bash

 | 2) SBERT -> Embeddings | - Embedding classifier
 +---------------------------+
 |
 +---------------------+----------------------+----------------+
 | | | |
 v v v v
 +------------+ +------------+ +-------------+
+-------------+
 | Keyword | | Recommender| | Classifier | |
Streamlit |
 | Exact Match| | (Nearest | | (Author ID) | | App
(UI) |
 | (regex) | | Neighbors) | | | | API /
Front |
 +------------+ +------------+ +-------------+
+-------------+
```

```bash
 \_________________Hybrid Search______________/ |
 (keyword-first, then semantic) v
 Hosting
(Streamlit)
Explanation - Raw data is cleaned and stored as poetry_data.csv .
- Two model pipelines are prepared: a fast TF-IDF baseline and a transformer embedding pipeline.
- The app exposes three retrieval modes: keyword, semantic, hybrid.
- Saved artifacts (.pkl / .npy) allow quick startup and avoid recomputing embeddings on every request
```