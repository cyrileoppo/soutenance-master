from __future__ import annotations

import streamlit as st

from services.bm25_service import build_bm25_index
from services.embedding_service import compute_corpus_embeddings
from services.model_loader import load_dataset


@st.cache_resource(show_spinner=False)
def get_cached_dataset(csv_path: str):
    return load_dataset(csv_path)


@st.cache_resource(show_spinner=False)
def get_cached_embeddings(model_path: str, csv_path: str):
    dataset = load_dataset(csv_path)
    return compute_corpus_embeddings(model_path=model_path, dataset=dataset)


@st.cache_resource(show_spinner=False)
def get_cached_bm25(csv_path: str):
    dataset = load_dataset(csv_path)
    return build_bm25_index(dataset)
