from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.model_loader import load_model
from utils.constants import ROOT_DIR


# Chemin du cache CSV contenant dataset + embeddings
EMBEDDINGS_CACHE_PATH = ROOT_DIR / 'data' / 'dataset_with_embeddings.csv'


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _embeddings_to_json_col(embeddings: np.ndarray) -> list[str]:
    """Convertit une matrice d'embeddings en liste de strings JSON pour stockage CSV."""
    return [json.dumps(row.tolist()) for row in embeddings]


def _json_col_to_embeddings(series: pd.Series) -> np.ndarray:
    """Convertit une colonne JSON en matrice numpy d'embeddings."""
    result = []
    for value in series:
        if isinstance(value, str):
            result.append(json.loads(value))
        elif isinstance(value, (int, float)):
            # Valeur corrompue/NaN - ne devrait pas arriver
            raise ValueError(
                f'Colonne d\'embedding corrompue (valeur numérique au lieu de JSON). '
                f'Supprimez data/dataset_with_embeddings.csv et relancez l\'app pour recalculer.'
            )
        else:
            result.append(json.loads(str(value)))
    return np.array(result, dtype=float)


def embeddings_cache_exists() -> bool:
    """Vérifie si le cache CSV avec embeddings existe et contient les colonnes nécessaires."""
    if not EMBEDDINGS_CACHE_PATH.exists():
        return False
    try:
        # Lire juste les premières lignes pour vérifier
        df = pd.read_csv(EMBEDDINGS_CACHE_PATH, nrows=2)
        required_cols = {'anchor_embedding', 'description_embedding', 'profile_embedding'}
        if not required_cols.issubset(set(df.columns)):
            return False
        # Vérifier que les colonnes contiennent bien du JSON (str commençant par '[')
        sample = df['anchor_embedding'].iloc[0]
        if not isinstance(sample, str) or not sample.startswith('['):
            return False
        return True
    except Exception:
        return False


def load_cached_embeddings_and_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge le dataset et les embeddings depuis le cache CSV."""
    df = pd.read_csv(EMBEDDINGS_CACHE_PATH)

    anchor_embeddings = _json_col_to_embeddings(df['anchor_embedding'])
    description_embeddings = _json_col_to_embeddings(df['description_embedding'])
    profile_embeddings = _json_col_to_embeddings(df['profile_embedding'])

    # Retirer les colonnes d'embeddings du dataset pour l'usage normal
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
    """Calcule les embeddings puis sauvegarde le tout dans un CSV pour les prochains lancements."""
    embedding_bundle = compute_corpus_embeddings(model_path, dataset)

    # Créer le CSV avec dataset + embeddings
    df_to_save = dataset.copy()
    df_to_save['anchor_embedding'] = _embeddings_to_json_col(embedding_bundle['anchor_embeddings'])
    df_to_save['description_embedding'] = _embeddings_to_json_col(embedding_bundle['description_embeddings'])
    df_to_save['profile_embedding'] = _embeddings_to_json_col(embedding_bundle['profile_embeddings'])

    EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_csv(EMBEDDINGS_CACHE_PATH, index=False)

    return embedding_bundle


def encode_query(model_path: str, text: str) -> np.ndarray:
    model = load_model(model_path)
    embedding = model.encode([text], normalize_embeddings=True)
    return np.asarray(embedding[0], dtype=float)
