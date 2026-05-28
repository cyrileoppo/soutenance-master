from __future__ import annotations

import numpy as np


def cosine_scores(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    return corpus_embeddings @ query_embedding


def cosine_similarity(left_embedding: np.ndarray, right_embedding: np.ndarray) -> float:
    return float(np.dot(left_embedding, right_embedding))


def vector_distance(score: float) -> float:
    return 1.0 - score
