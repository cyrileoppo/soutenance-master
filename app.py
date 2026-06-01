from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer

from functions import (
    ASSETS_PATH,
    MODEL_PATH,
    build_bm25_index,
    compute_corpus_embeddings,
    download_kaggle_dataset,
    embeddings_cache_exists,
    ensure_model_available,
    load_cached_embeddings_and_dataset,
    prepare_dataset,
    save_embeddings_to_cache,
)
from views import (
    embedding_space_page,
    limitations_page,
    semantic_search_page,
    semantic_vs_lexical_page,
    similarity_analyzer_page,
)


PAGES = {
    'Recherche s\u00e9mantique': semantic_search_page,
    'BM25 vs S\u00e9mantique': semantic_vs_lexical_page,
    'Espace d\u2019embeddings': embedding_space_page,
    'Analyseur de similarit\u00e9': similarity_analyzer_page,
    'Limites du mod\u00e8le': limitations_page,
}


def inject_styles() -> None:
    st.markdown(f'<style>{ASSETS_PATH.read_text()}</style>', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Chargement du mod\u00e8le fine-tun\u00e9...')
def load_model(model_path: str) -> SentenceTransformer:
    resolved_path = ensure_model_available(model_path)
    return SentenceTransformer(resolved_path)


@st.cache_resource(show_spinner='Chargement du dataset depuis Kaggle...')
def load_dataset():
    import pandas as pd
    csv_path = download_kaggle_dataset()
    df = pd.read_csv(csv_path)
    return prepare_dataset(df)


@st.cache_resource(show_spinner='Chargement des donn\u00e9es et embeddings...')
def get_cached_dataset_and_embeddings(model_path: str):
    if embeddings_cache_exists():
        st.info('Chargement depuis le cache local (data/dataset_with_embeddings.parquet)')
        return load_cached_embeddings_and_dataset()

    st.info('Premi\u00e8re ex\u00e9cution : calcul des embeddings (sera mis en cache pour la suite)...')
    dataset = load_dataset()
    model = load_model(model_path)
    embedding_bundle = compute_corpus_embeddings(model, dataset)
    save_embeddings_to_cache(dataset, embedding_bundle)
    return dataset, embedding_bundle


@st.cache_resource(show_spinner='Construction de l\u2019index BM25...')
def get_cached_bm25(_dataset):
    return build_bm25_index(_dataset)


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
