from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.model_loader import load_model
from utils.constants import ROOT_DIR


# Chemin du cache Parquet contenant dataset + embeddings
EMBEDDINGS_CACHE_PATH = ROOT_DIR / 'data' / 'dataset_with_embeddings.parquet'


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embeddings_cache_exists() -> bool:
    """Vérifie si le cache Parquet avec embeddings existe."""
    if not EMBEDDINGS_CACHE_PATH.exists():
        return False
    try:
        df = pd.read_parquet(EMBEDDINGS_CACHE_PATH, columns=['anchor_embedding'])
        return len(df) > 0
    except Exception:
        return False


def load_cached_embeddings_and_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge le dataset et les embeddings depuis le cache Parquet."""
    df = pd.read_parquet(EMBEDDINGS_CACHE_PATH)

    # Les colonnes d'embeddings sont stockées comme listes Python dans Parquet
    anchor_embeddings = np.array(df['anchor_embedding'].tolist(), dtype=float)
    description_embeddings = np.array(df['description_embedding'].tolist(), dtype=float)
    profile_embeddings = np.array(df['profile_embedding'].tolist(), dtype=float)

    dataset = df.drop(columns=['anchor_embedding', 'description_embedding', 'profile_embedding'])

    embedding_bundle = {
        'anchor_embeddings': anchor_embeddings,
        'description_embeddings': description_embeddings,
        'profile_embeddings': profile_embeddings,
    }
    return dataset, embedding_bundle


def compute_corpus_embeddings(model_path: str, dataset: pd.DataFrame) -> dict[str, Any]:
    """Calcule les embeddings du corpus et les retourne."""
    model = load_model(model_path)
    anchor_embeddings = model.encode(dataset['anchor_text'].tolist(), normalize_embeddings=True)
    description_embeddings = model.encode(dataset['description_text'].tolist(), normalize_embeddings=True)
    profile_embeddings = _normalize_rows((anchor_embeddings + description_embeddings) / 2.0)
    return {
        'anchor_embeddings': np.asarray(anchor_embeddings, dtype=float),
        'description_embeddings': np.asarray(description_embeddings, dtype=float),
        'profile_embeddings': np.asarray(profile_embeddings, dtype=float),
    }


def compute_and_save_embeddings(model_path: str, dataset: pd.DataFrame) -> dict[str, Any]:
    """Calcule les embeddings puis sauvegarde le tout en Parquet pour les prochains lancements."""
    embedding_bundle = compute_corpus_embeddings(model_path, dataset)

    df_to_save = dataset.copy()
    # Stocker les embeddings comme listes Python (Parquet les gère nativement)
    df_to_save['anchor_embedding'] = list(embedding_bundle['anchor_embeddings'])
    df_to_save['description_embedding'] = list(embedding_bundle['description_embeddings'])
    df_to_save['profile_embedding'] = list(embedding_bundle['profile_embeddings'])

    EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_parquet(EMBEDDINGS_CACHE_PATH, index=False)

    return embedding_bundle


def encode_query(model_path: str, text: str) -> np.ndarray:
    model = load_model(model_path)
    embedding = model.encode([text], normalize_embeddings=True)
    return np.asarray(embedding[0], dtype=float)
