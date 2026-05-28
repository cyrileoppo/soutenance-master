from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from services.retrieval_service import compare_listings
from services.similarity_service import vector_distance
from utils.helpers import similarity_verdict


def render(dataset, embedding_bundle) -> None:
    st.title('Live Similarity Analyzer')
    st.caption('Compare two listings live and inspect a calibrated semantic similarity score derived from the trained anchor/description geometry.')

    left_col, right_col = st.columns(2)
    with left_col:
        left_title = st.selectbox('Listing A', dataset['title'].tolist(), index=0)
    with right_col:
        right_title = st.selectbox('Listing B', dataset['title'].tolist(), index=5)

    left_row = dataset.loc[dataset['title'] == left_title].iloc[0]
    right_row = dataset.loc[dataset['title'] == right_title].iloc[0]

    score = compare_listings(
        anchor_embeddings=embedding_bundle['anchor_embeddings'],
        description_embeddings=embedding_bundle['description_embeddings'],
        left_listing_id=int(left_row['listing_id']),
        right_listing_id=int(right_row['listing_id']),
    )
    distance = vector_distance(score)
    verdict = similarity_verdict(score)

    metric_col, gauge_col = st.columns([1, 1.2])
    with metric_col:
        st.metric('Cosine similarity', f'{score:.3f}')
        st.metric('Vector distance', f'{distance:.3f}')
        st.markdown(f"<div class='verdict-pill'>{verdict}</div>", unsafe_allow_html=True)
    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=score,
            number={'valueformat': '.3f'},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#60a5fa'},
                'steps': [
                    {'range': [0, 0.65], 'color': '#1f2937'},
                    {'range': [0.65, 0.85], 'color': '#374151'},
                    {'range': [0.85, 1], 'color': '#0f766e'},
                ],
            },
        ))
        fig.update_layout(template='plotly_dark', margin={'l': 10, 'r': 10, 't': 20, 'b': 0}, height=320)
        st.plotly_chart(fig, use_container_width=True)

    compare_left, compare_right = st.columns(2)
    with compare_left:
        st.markdown(f"### {left_row['title']}")
        st.caption(f"{left_row['propertyType']} • {left_row['bedrooms']} bed • {left_row['price']}")
        st.write(left_row['clean_description'])
    with compare_right:
        st.markdown(f"### {right_row['title']}")
        st.caption(f"{right_row['propertyType']} • {right_row['bedrooms']} bed • {right_row['price']}")
        st.write(right_row['clean_description'])
