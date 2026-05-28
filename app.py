from __future__ import annotations

import streamlit as st

from services.model_loader import load_dataset
from utils.constants import ASSETS_PATH, COLOR_OPTIONS, DATA_PATH, MODEL_PATH
from utils.cache import get_cached_bm25, get_cached_embeddings
from views import embedding_space_page, limitations_page, semantic_search_page, semantic_vs_lexical_page, similarity_analyzer_page


PAGES = {
    'Semantic Property Retrieval': semantic_search_page,
    'BM25 vs Semantic Retrieval': semantic_vs_lexical_page,
    'Embedding Space Explorer': embedding_space_page,
    'Live Similarity Analyzer': similarity_analyzer_page,
    'Model Limitations': limitations_page,
}


def inject_styles() -> None:
    st.markdown(f'<style>{ASSETS_PATH.read_text()}</style>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title='London Real Estate Semantic Search Demo', page_icon='🏙️', layout='wide')
    inject_styles()

    st.sidebar.title('Master Thesis Demo')
    st.sidebar.caption('Fine-tuned BGE-base-en-v1.5 bi-encoder for semantic property retrieval in London real estate.')
    model_path = st.sidebar.text_input('Mounted Google Drive model path', value=MODEL_PATH)
    page_name = st.sidebar.radio('Navigate', list(PAGES.keys()))
    st.sidebar.info('The app expects a locally mounted Google Drive path that can be passed directly to SentenceTransformer(MODEL_PATH).')

    dataset = load_dataset(str(DATA_PATH))

    try:
        embedding_bundle = get_cached_embeddings(model_path, str(DATA_PATH))
        bm25_index = get_cached_bm25(str(DATA_PATH))
    except Exception as exc:  # pragma: no cover - Streamlit runtime branch
        st.error('The model could not be loaded from the configured Google Drive path.')
        st.exception(exc)
        st.stop()

    page = PAGES[page_name]
    if page_name == 'Semantic Property Retrieval':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, model_path=model_path)
    elif page_name == 'BM25 vs Semantic Retrieval':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, bm25_index=bm25_index, model_path=model_path)
    elif page_name == 'Embedding Space Explorer':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle)
    elif page_name == 'Live Similarity Analyzer':
        page.render(dataset=dataset, embedding_bundle=embedding_bundle)
    else:
        page.render(dataset=dataset, embedding_bundle=embedding_bundle, model_path=model_path)


if __name__ == '__main__':
    main()
