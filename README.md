# Streamlit Demo Application for a Master's Thesis Defense

This repository contains a Streamlit demonstration for a Master 2 Data & AI thesis on semantic retrieval for London real estate listings.

## Features

- exact training-time preprocessing with `[TITLE]`, `[DESC]`, and `[ATTR]`
- semantic property retrieval from a selected listing
- BM25 vs semantic retrieval comparison
- interactive UMAP embedding space explorer
- live listing-to-listing similarity analyzer
- explicit limitations page for off-domain and non-English examples

## Project structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   └── README.md
├── data/
│   └── sample_listings.csv
├── services/
│   ├── model_loader.py
│   ├── preprocessing_service.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── similarity_service.py
│   ├── bm25_service.py
│   └── visualization_service.py
├── views/
│   ├── semantic_search_page.py
│   ├── semantic_vs_lexical_page.py
│   ├── embedding_space_page.py
│   ├── similarity_analyzer_page.py
│   └── limitations_page.py
├── components/
│   ├── result_card.py
│   ├── similarity_bar.py
│   ├── comparison_table.py
│   ├── embedding_plot.py
│   └── loading_spinner.py
├── utils/
│   ├── constants.py
│   ├── cache.py
│   └── helpers.py
└── assets/
    └── styles.css
```

## Model path

The fine-tuned model is intentionally not versioned in Git. Configure a locally mounted or synchronized Google Drive path in either of these ways:

- edit `utils/constants.py`
- or export `MODEL_PATH` before launching the app
- or override the default path from the Streamlit sidebar at runtime

Expected examples:

- `/content/drive/MyDrive/final_model`
- `G:/My Drive/final_model`
- `/Users/name/Library/CloudStorage/GoogleDrive/final_model`

The application loads the model directly with `SentenceTransformer(MODEL_PATH)`.

## Installation

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
streamlit run app.py
```

## Preprocessing contract

The preprocessing logic matches the training notebook:

- clean HTML descriptions with BeautifulSoup
- fill `title`, `propertyType`, `sizeSqFeetMax`, `bedrooms`, `bathrooms`, and `price` with `N/A`
- strip trailing `.0` from bedroom and bathroom values
- build a `rich_anchor` exactly as:

```text
title + " [ATTR] Type: " + propertyType + " [ATTR] Beds: " + bedrooms + " [ATTR] Baths: " + bathrooms + " [ATTR] Size: " + sizeSqFeetMax + " sqft" + " [ATTR] Price: " + price
```

Encoding uses:

- `"[TITLE] " + rich_anchor`
- `"[DESC] " + clean_description`

## Notes for the defense

- the dataset in `data/sample_listings.csv` is synthetic but realistic and intentionally lightweight for a smooth live demo
- embeddings are pre-computed and cached in memory for instant interactions
- limitations are shown deliberately to explain domain specialization, distribution shift, and lack of multilingual fine-tuning
