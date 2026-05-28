from __future__ import annotations

import streamlit as st


def render_similarity_bar(score: float, label: str = 'Cosine similarity') -> None:
    bounded_score = max(0.0, min(1.0, score))
    st.markdown(f"<div class='metric-caption'>{label}</div>", unsafe_allow_html=True)
    st.progress(bounded_score)
    st.caption(f'{bounded_score:.3f}')
