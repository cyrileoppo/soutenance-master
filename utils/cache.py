from __future__ import annotations

import streamlit as st

from services.bm25_service import build_bm25_index
from services.embedding_service import compute_corpus_embeddings
from services.model_loader import load_dataset


@st.cache_resource(show_spinner='Chargement du dataset...')
def get_cached_dataset():
    return load_dataset()


@st.cache_resource(show_spinner='Calcul des embeddings du corpus...')
def get_cached_embeddings(model_path: str):
    dataset = load_dataset()
    return compute_corpus_embeddings(model_path=model_path, dataset=dataset)


@st.cache_resource(show_spinner='Construction de l\u2019index BM25...')
def get_cached_bm25():
    dataset = load_dataset()
    return build_bm25_index(dataset)
