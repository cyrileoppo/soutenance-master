from __future__ import annotations

import streamlit as st

from components.result_card import render_result_card
from services.retrieval_service import semantic_search
from utils.constants import LIMITATION_QUERIES, TOP_K


@st.cache_data(show_spinner=False)
def _run_limitation_searches(_dataset, _description_embeddings, model_path: str) -> list:
    """Cache les resultats des recherches de la page Limites pour eviter le recalcul a chaque rendu."""
    results = []
    for example in LIMITATION_QUERIES:
        result = semantic_search(
            model_path=model_path,
            dataset=_dataset,
            description_embeddings=_description_embeddings,
            query_text=example['query_text'],
            top_k=TOP_K,
        )
        max_similarity = float(result['similarity'].max()) if not result.empty else 0.0
        results.append({'result': result, 'max_similarity': max_similarity})
    return results


def render(dataset, embedding_bundle, model_path: str) -> None:
    st.title('Limites du Mod\u00e8le')
    st.caption('Ces cas d\u2019\u00e9chec contr\u00f4l\u00e9s montrent pourquoi l\u2019encodeur est sp\u00e9cialis\u00e9 sur un domaine, anglophone uniquement, et n\u2019est pas un LLM multilingue g\u00e9n\u00e9raliste.')

    cached_results = _run_limitation_searches(dataset, embedding_bundle['description_embeddings'], model_path)

    for example, cached in zip(LIMITATION_QUERIES, cached_results):
        with st.container(border=True):
            st.markdown(f"### {example['label']}")
            st.code(example['query_text'])
            st.metric('Meilleure similarit\u00e9 cosinus', f"{cached['max_similarity']:.3f}")
            st.write(example['explanation'])
            for rank, (_, row) in enumerate(cached['result'].iterrows(), start=1):
                render_result_card(row, rank)

    st.markdown(
        """
        ### Explication scientifique
        - **Sp\u00e9cialisation de domaine** : le fine-tuning a orient\u00e9 l\u2019encodeur vers la s\u00e9mantique de l\u2019immobilier r\u00e9sidentiel londonien.
        - **D\u00e9calage de distribution** : la g\u00e9ographie parisienne, les conventions de prix et la syntaxe fran\u00e7aise \u00e9loignent la requ\u00eate de la distribution d\u2019entra\u00eenement.
        - **Pas de support multilingue** : le mod\u00e8le a \u00e9t\u00e9 optimis\u00e9 uniquement sur des annonces en anglais, le transfert cross-lingue est donc faible et peu fiable.
        - **Ce n\u2019est pas un LLM** : il projette les textes en vecteurs pour la recherche ; il ne raisonne pas, ne traduit pas et ne g\u00e9n\u00e8re pas d\u2019explications de mani\u00e8re autonome.
        """
    )
