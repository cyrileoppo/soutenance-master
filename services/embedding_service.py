from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.model_loader import load_model


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def compute_corpus_embeddings(model_path: str, dataset: pd.DataFrame) -> dict[str, Any]:
    model = load_model(model_path)
    anchor_embeddings = model.encode(dataset['anchor_text'].tolist(), normalize_embeddings=True)
    description_embeddings = model.encode(dataset['description_text'].tolist(), normalize_embeddings=True)
    profile_embeddings = _normalize_rows((anchor_embeddings + description_embeddings) / 2.0)
    return {
        'anchor_embeddings': np.asarray(anchor_embeddings, dtype=float),
        'description_embeddings': np.asarray(description_embeddings, dtype=float),
        'profile_embeddings': np.asarray(profile_embeddings, dtype=float),
    }


def encode_query(model_path: str, text: str) -> np.ndarray:
    model = load_model(model_path)
    embedding = model.encode([text], normalize_embeddings=True)
    return np.asarray(embedding[0], dtype=float)
