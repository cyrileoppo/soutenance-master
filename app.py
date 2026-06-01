from __future__ import annotations

import streamlit as st

from utils.constants import ASSETS_PATH, MODEL_PATH
from utils.cache import get_cached_bm25, get_cached_dataset_and_embeddings
from views import embedding_space_page, limitations_page, semantic_search_page, semantic_vs_lexical_page, similarity_analyzer_page


PAGES = {
    'Recherche sémantique': semantic_search_page,
    'BM25 vs Sémantique': semantic_vs_lexical_page,
    'Espace d\u2019embeddings': embedding_space_page,
    'Analyseur de similarité': similarity_analyzer_page,
    'Limites du modèle': limitations_page,
}


def inject_styles() -> None:
    st.markdown(f'<style>{ASSETS_PATH.read_text()}</style>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title='Recherche Sémantique Immobilière - Soutenance Master', page_icon='\U0001f3d9\ufe0f', layout='wide')
    inject_styles()

    st.sidebar.title('Soutenance Master 2')
    st.sidebar.caption('Bi-encodeur BGE-base-en-v1.5 fine-tuné pour la recherche sémantique d\u2019annonces immobilières londoniennes.')
    model_path = st.sidebar.text_input('Chemin du modèle (Google Drive)', value=MODEL_PATH)
    page_name = st.sidebar.radio('Navigation', list(PAGES.keys()))
    st.sidebar.info('L\u2019application charge le modèle depuis un chemin Google Drive local. Vous pouvez aussi définir la variable d\u2019environnement MODEL_PATH.')

    try:
        dataset, embedding_bundle = get_cached_dataset_and_embeddings(model_path)
        bm25_index = get_cached_bm25(dataset)
    except Exception as exc:
        st.error('Erreur lors du chargement des données ou du modèle.')
        st.exception(exc)
        st.stop()

    page = PAGES[page_name]
    if page_name == 'Recherche sémantique':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, model_path=model_path)
    elif page_name == 'BM25 vs Sémantique':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, bm25_index=bm25_index, model_path=model_path)
    elif page_name == 'Espace d\u2019embeddings':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle)
    elif page_name == 'Analyseur de similarité':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle)
    else:
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, model_path=model_path)


if __name__ == '__main__':
    main()
