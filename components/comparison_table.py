from __future__ import annotations

import pandas as pd
import streamlit as st


def render_comparison_table(bm25_results: pd.DataFrame, semantic_results: pd.DataFrame) -> None:
    bm25_table = bm25_results[['title', 'propertyType', 'bedrooms', 'bm25_score']].copy()
    semantic_table = semantic_results[['title', 'propertyType', 'bedrooms', 'similarity']].copy()
    bm25_table['bm25_score'] = bm25_table['bm25_score'].map(lambda value: f'{value:.2f}')
    semantic_table['similarity'] = semantic_table['similarity'].map(lambda value: f'{value:.3f}')
    left, right = st.columns(2)
    with left:
        st.markdown('### Recherche lexicale BM25')
        st.dataframe(bm25_table, use_container_width=True, hide_index=True)
    with right:
        st.markdown('### Recherche sémantique')
        st.dataframe(semantic_table, use_container_width=True, hide_index=True)
