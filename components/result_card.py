from __future__ import annotations

import pandas as pd
import streamlit as st

from components.similarity_bar import render_similarity_bar


def render_result_card(row: pd.Series, rank: int, score_column: str = 'similarity') -> None:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class='result-card-header'>
                <div>
                    <div class='result-rank'>#{rank}</div>
                    <h4>{row['title']}</h4>
                </div>
                <div class='result-badges'>
                    <span>{row['propertyType']}</span>
                    <span>{row['bedrooms']} bed</span>
                    <span>{row['bathrooms']} bath</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(row['clean_description'])
        render_similarity_bar(float(row[score_column]), 'Cosine similarity' if score_column == 'similarity' else 'BM25 score (rescaled)')
        st.caption(f"{row['sizeSqFeetMax']} sqft • {row['price']}")
