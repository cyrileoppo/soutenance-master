from __future__ import annotations

import streamlit as st

from components.loading_spinner import render_pipeline
from components.result_card import render_result_card
from services.retrieval_service import semantic_search
from utils.constants import PIPELINE_STAGES, TOP_K


def render(dataset, embedding_bundle, model_path: str) -> None:
    st.title('Recherche Sémantique d\u2019Annonces')
    st.caption('Sélectionnez une annonce londonienne existante, encodez-la avec le bi-encodeur fine-tuné et retrouvez instantanément les voisins sémantiques les plus proches.')

    options = dataset['title'].tolist()
    selected_title = st.selectbox('Choisir une annonce de référence', options, index=0)
    if st.button('Annonce aléatoire \U0001f3b2'):
        selected_title = dataset.sample(1, random_state=None)['title'].iloc[0]
        st.session_state['semantic_selected_title'] = selected_title
    selected_title = st.session_state.get('semantic_selected_title', selected_title)
    st.session_state['semantic_selected_title'] = selected_title

    selected_row = dataset.loc[dataset['title'] == selected_title].iloc[0]

    with st.container(border=True):
        st.markdown(f"### {selected_row['title']}")
        st.caption(f"{selected_row['propertyType']} \u2022 {selected_row['bedrooms']} ch. \u2022 {selected_row['bathrooms']} sdb \u2022 {selected_row['sizeSqFeetMax']} sqft \u2022 {selected_row['price']}")
        st.write(selected_row['clean_description'])

    if st.button('Trouver les correspondances sémantiques', type='primary'):
        render_pipeline(PIPELINE_STAGES)
        results = semantic_search(
            model_path=model_path,
            dataset=dataset,
            description_embeddings=embedding_bundle['description_embeddings'],
            query_text=selected_row['anchor_text'],
            top_k=TOP_K,
            exclude_listing_id=int(selected_row['listing_id']),
        )
        st.markdown('### Meilleures correspondances sémantiques')
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            render_result_card(row, rank)
