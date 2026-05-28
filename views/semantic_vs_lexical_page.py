from __future__ import annotations

import streamlit as st

from components.comparison_table import render_comparison_table
from components.result_card import render_result_card
from services.bm25_service import bm25_search
from services.retrieval_service import semantic_search
from utils.constants import TOP_K


def render(dataset, embedding_bundle, bm25_index, model_path: str) -> None:
    st.title('BM25 vs Semantic Retrieval')
    st.caption('The same listing is used as a query for both retrieval systems so the jury can immediately compare lexical overlap with semantic understanding.')

    selected_title = st.selectbox('Comparison query listing', dataset['title'].tolist(), index=1)
    selected_row = dataset.loc[dataset['title'] == selected_title].iloc[0]

    st.markdown(f"**Query anchor**  \n`{selected_row['rich_anchor']}`")

    bm25_results = bm25_search(
        bm25_index=bm25_index,
        dataset=dataset,
        query_text=selected_row['rich_anchor'],
        top_k=TOP_K,
        exclude_listing_id=int(selected_row['listing_id']),
    )
    semantic_results = semantic_search(
        model_path=model_path,
        dataset=dataset,
        description_embeddings=embedding_bundle['description_embeddings'],
        query_text=selected_row['anchor_text'],
        top_k=TOP_K,
        exclude_listing_id=int(selected_row['listing_id']),
    )

    render_comparison_table(bm25_results, semantic_results)

    left, right = st.columns(2)
    with left:
        for rank, (_, row) in enumerate(bm25_results.iterrows(), start=1):
            normalized_score = min(float(row['bm25_score']) / max(float(bm25_results['bm25_score'].max()), 1e-6), 1.0)
            row = row.copy()
            row['bm25_score'] = normalized_score
            render_result_card(row, rank, score_column='bm25_score')
    with right:
        for rank, (_, row) in enumerate(semantic_results.iterrows(), start=1):
            render_result_card(row, rank)
