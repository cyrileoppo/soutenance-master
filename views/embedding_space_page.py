from __future__ import annotations

import streamlit as st

from components.embedding_plot import render_embedding_plot
from components.result_card import render_result_card
from services.retrieval_service import nearest_neighbors
from services.visualization_service import build_embedding_figure, compute_projection
from utils.constants import COLOR_OPTIONS, TOP_K


def render(dataset, embedding_bundle) -> None:
    st.title('Explorateur de l\u2019Espace d\u2019Embeddings')
    st.caption('UMAP révèle comment l\u2019encodeur fine-tuné organise les annonces londoniennes dans un espace sémantique de faible dimension.')

    color_label = st.selectbox('Colorer les points par', list(COLOR_OPTIONS.keys()))
    projection = compute_projection(dataset, embedding_bundle['profile_embeddings'], COLOR_OPTIONS[color_label])
    fig = build_embedding_figure(projection)
    selection = render_embedding_plot(fig)

    selected_listing_id = None
    if selection and selection.get('selection', {}).get('points'):
        point = selection['selection']['points'][0]
        custom_data = point.get('customdata', [])
        if custom_data:
            selected_listing_id = int(custom_data[0])

    fallback_title = st.selectbox('Ou inspecter une annonce directement', dataset['title'].tolist(), index=2)
    if selected_listing_id is None:
        selected_listing_id = int(dataset.loc[dataset['title'] == fallback_title, 'listing_id'].iloc[0])

    selected_row = dataset.loc[dataset['listing_id'] == selected_listing_id].iloc[0]
    st.markdown('### Annonce sélectionnée')
    selected_display_row = selected_row.copy()
    selected_display_row['similarity'] = 1.0
    render_result_card(selected_display_row, rank=0)

    neighbours = nearest_neighbors(dataset, embedding_bundle['profile_embeddings'], selected_listing_id, TOP_K)
    st.markdown('### Voisins les plus proches dans l\u2019espace d\u2019embeddings')
    for rank, (_, row) in enumerate(neighbours.iterrows(), start=1):
        render_result_card(row, rank)
