from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.embedding_service import encode_query
from services.similarity_service import cosine_scores, cosine_similarity


def semantic_search(
    model_path: str,
    dataset: pd.DataFrame,
    description_embeddings: np.ndarray,
    query_text: str,
    top_k: int,
    exclude_listing_id: int | None = None,
) -> pd.DataFrame:
    query_embedding = encode_query(model_path, query_text)
    scores = cosine_scores(query_embedding, description_embeddings)
    ranked = dataset.copy()
    ranked['similarity'] = scores
    if exclude_listing_id is not None:
        ranked = ranked[ranked['listing_id'] != exclude_listing_id]
    return ranked.sort_values('similarity', ascending=False).head(top_k).reset_index(drop=True)


def nearest_neighbors(
    dataset: pd.DataFrame,
    profile_embeddings: np.ndarray,
    listing_id: int,
    top_k: int,
) -> pd.DataFrame:
    query_embedding = profile_embeddings[listing_id]
    scores = cosine_scores(query_embedding, profile_embeddings)
    ranked = dataset.copy()
    ranked['similarity'] = scores
    ranked = ranked[ranked['listing_id'] != listing_id]
    return ranked.sort_values('similarity', ascending=False).head(top_k).reset_index(drop=True)


def compare_listings(
    anchor_embeddings: np.ndarray,
    description_embeddings: np.ndarray,
    left_listing_id: int,
    right_listing_id: int,
) -> float:
    left_to_right = cosine_similarity(anchor_embeddings[left_listing_id], description_embeddings[right_listing_id])
    right_to_left = cosine_similarity(anchor_embeddings[right_listing_id], description_embeddings[left_listing_id])
    return float((left_to_right + right_to_left) / 2.0)
