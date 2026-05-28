from __future__ import annotations

import streamlit as st

from components.result_card import render_result_card
from services.retrieval_service import semantic_search
from utils.constants import LIMITATION_QUERIES, TOP_K


def render(dataset, embedding_bundle, model_path: str) -> None:
    st.title('Model Limitations')
    st.caption('These controlled failure cases show why the encoder is domain-specialized, English-only for this task, and not a general-purpose multilingual LLM.')

    for example in LIMITATION_QUERIES:
        with st.container(border=True):
            st.markdown(f"### {example['label']}")
            st.code(example['query_text'])
            results = semantic_search(
                model_path=model_path,
                dataset=dataset,
                description_embeddings=embedding_bundle['description_embeddings'],
                query_text=example['query_text'],
                top_k=TOP_K,
            )
            max_similarity = float(results['similarity'].max()) if not results.empty else 0.0
            st.metric('Best cosine similarity', f'{max_similarity:.3f}')
            st.write(example['explanation'])
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                render_result_card(row, rank)

    st.markdown(
        """
        ### Scientific explanation
        - **Domain specialization**: fine-tuning pushed the encoder toward London residential property semantics.
        - **Distribution shift**: Parisian geography, pricing conventions, and French syntax move the query away from the training distribution.
        - **No multilingual support in this setup**: the model was optimized on English-only listings, so cross-lingual transfer is weak and unreliable.
        - **Not an LLM**: it maps texts into vectors for retrieval; it does not reason, translate, or generate explanations autonomously.
        """
    )
