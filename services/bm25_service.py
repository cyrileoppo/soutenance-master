from __future__ import annotations

from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(dataset: pd.DataFrame) -> dict[str, Any]:
    corpus = dataset['clean_description'].tolist()
    tokenized_corpus = [_tokenize(document) for document in corpus]
    return {'bm25': BM25Okapi(tokenized_corpus), 'corpus': corpus}


def bm25_search(bm25_index: dict[str, Any], dataset: pd.DataFrame, query_text: str, top_k: int, exclude_listing_id: int | None = None) -> pd.DataFrame:
    scores = bm25_index['bm25'].get_scores(_tokenize(query_text))
    ranked = dataset.copy()
    ranked['bm25_score'] = scores
    if exclude_listing_id is not None:
        ranked = ranked[ranked['listing_id'] != exclude_listing_id]
    return ranked.sort_values('bm25_score', ascending=False).head(top_k).reset_index(drop=True)
