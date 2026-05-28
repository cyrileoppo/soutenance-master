from __future__ import annotations

import streamlit as st

from components.loading_spinner import render_pipeline
from components.result_card import render_result_card
from services.retrieval_service import semantic_search
from utils.constants import PIPELINE_STAGES, TOP_K


def render(dataset, embedding_bundle, model_path: str) -> None:
    st.title('Semantic Property Retrieval')
    st.caption('Select an existing London listing, encode it with the fine-tuned bi-encoder, and retrieve the nearest semantic neighbours instantly.')

    options = dataset['title'].tolist()
    selected_title = st.selectbox('Choose a reference listing', options, index=0)
    if st.button('Pick a random listing'):
        selected_title = dataset.sample(1, random_state=None)['title'].iloc[0]
        st.session_state['semantic_selected_title'] = selected_title
    selected_title = st.session_state.get('semantic_selected_title', selected_title)
    st.session_state['semantic_selected_title'] = selected_title

    selected_row = dataset.loc[dataset['title'] == selected_title].iloc[0]

    with st.container(border=True):
        st.markdown(f"### {selected_row['title']}")
        st.caption(f"{selected_row['propertyType']} • {selected_row['bedrooms']} bed • {selected_row['bathrooms']} bath • {selected_row['sizeSqFeetMax']} sqft • {selected_row['price']}")
        st.write(selected_row['clean_description'])

    if st.button('Find Semantic Matches', type='primary'):
        render_pipeline(PIPELINE_STAGES)
        results = semantic_search(
            model_path=model_path,
            dataset=dataset,
            description_embeddings=embedding_bundle['description_embeddings'],
            query_text=selected_row['anchor_text'],
            top_k=TOP_K,
            exclude_listing_id=int(selected_row['listing_id']),
        )
        st.markdown('### Top semantic matches')
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            render_result_card(row, rank)
