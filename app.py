from __future__ import annotations

import streamlit as st

from utils.constants import ASSETS_PATH, MODEL_PATH
from utils.cache import get_cached_bm25, get_cached_dataset_and_embeddings
from views import embedding_space_page, limitations_page, semantic_search_page, semantic_vs_lexical_page, similarity_analyzer_page


PAGES = {
    'Recherche s\u00e9mantique': semantic_search_page,
    'BM25 vs S\u00e9mantique': semantic_vs_lexical_page,
    'Espace d\u2019embeddings': embedding_space_page,
    'Analyseur de similarit\u00e9': similarity_analyzer_page,
    'Limites du mod\u00e8le': limitations_page,
}


def inject_styles() -> None:
    """Injecte le CSS custom. Si le fichier est absent, ignore silencieusement."""
    if ASSETS_PATH.exists():
        st.markdown(f'<style>{ASSETS_PATH.read_text()}</style>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title='Recherche S\u00e9mantique Immobili\u00e8re - Soutenance Master', page_icon='\U0001f3d9\ufe0f', layout='wide')
    inject_styles()

    st.sidebar.title('Soutenance Master 2')
    st.sidebar.caption('Bi-encodeur BGE-base-en-v1.5 fine-tun\u00e9 pour la recherche s\u00e9mantique d\u2019annonces immobili\u00e8res londoniennes.')
    model_path = st.sidebar.text_input('Chemin du mod\u00e8le (Google Drive)', value=MODEL_PATH)
    page_name = st.sidebar.radio('Navigation', list(PAGES.keys()))
    st.sidebar.info('L\u2019application charge le mod\u00e8le depuis un chemin Google Drive local. Vous pouvez aussi d\u00e9finir la variable d\u2019environnement MODEL_PATH.')

    try:
        dataset, embedding_bundle = get_cached_dataset_and_embeddings(model_path)
        bm25_index = get_cached_bm25(dataset)
    except Exception as exc:
        st.error('Erreur lors du chargement des donn\u00e9es ou du mod\u00e8le.')
        st.exception(exc)
        st.stop()

    page = PAGES[page_name]
    if page_name == 'Recherche s\u00e9mantique':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, model_path=model_path)
    elif page_name == 'BM25 vs S\u00e9mantique':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, bm25_index=bm25_index, model_path=model_path)
    elif page_name == 'Espace d\u2019embeddings':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle)
    elif page_name == 'Analyseur de similarit\u00e9':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle)
    else:
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, model_path=model_path)


if __name__ == '__main__':
    main()
