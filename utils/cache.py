from __future__ import annotations

import streamlit as st

from services.bm25_service import build_bm25_index
from services.embedding_service import (
    compute_and_save_embeddings,
    embeddings_cache_exists,
    load_cached_embeddings_and_dataset,
)
from services.model_loader import load_dataset


@st.cache_resource(show_spinner='Chargement des données et embeddings...')
def get_cached_dataset_and_embeddings(model_path: str):
    """Charge le dataset et les embeddings.

    Si le fichier data/dataset_with_embeddings.csv existe et contient
    les embeddings, les charge directement (pas besoin du modèle).
    Sinon, télécharge le dataset Kaggle, calcule les embeddings,
    et sauvegarde le tout dans le CSV pour les prochains lancements.
    """
    if embeddings_cache_exists():
        st.info('✅ Chargement depuis le cache local (data/dataset_with_embeddings.csv)')
        dataset, embedding_bundle = load_cached_embeddings_and_dataset()
        return dataset, embedding_bundle

    # Pas de cache : charger le dataset et calculer les embeddings
    st.info('🔄 Première exécution : calcul des embeddings (sera mis en cache pour la suite)...')
    dataset = load_dataset()
    embedding_bundle = compute_and_save_embeddings(model_path, dataset)
    return dataset, embedding_bundle


@st.cache_resource(show_spinner='Construction de l\u2019index BM25...')
def get_cached_bm25(_dataset):
    """Construit l'index BM25 à partir du dataset."""
    return build_bm25_index(_dataset)
