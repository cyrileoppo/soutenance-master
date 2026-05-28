from __future__ import annotations

import streamlit as st

from components.result_card import render_result_card
from services.retrieval_service import semantic_search
from utils.constants import LIMITATION_QUERIES, TOP_K


def render(dataset, embedding_bundle, model_path: str) -> None:
    st.title('Limites du Modèle')
    st.caption('Ces cas d\u2019échec contrôlés montrent pourquoi l\u2019encodeur est spécialisé sur un domaine, anglophone uniquement, et n\u2019est pas un LLM multilingue généraliste.')

    for example in LIMITATION_QUERIES:
        with st.container(border=True):
            st.markdown(f"### {example['label']}")
            st.code(example['query_text'])
            results = semantic_search(
                model_path=model_path,
                dataset=dataset,
                description_embeddings=embedding_bundle['description_embeddings'],
                query_text=example['query_text'],
                top_k=TOP_K,
            )
            max_similarity = float(results['similarity'].max()) if not results.empty else 0.0
            st.metric('Meilleure similarité cosinus', f'{max_similarity:.3f}')
            st.write(example['explanation'])
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                render_result_card(row, rank)

    st.markdown(
        """
        ### Explication scientifique
        - **Spécialisation de domaine** : le fine-tuning a orienté l\u2019encodeur vers la sémantique de l\u2019immobilier résidentiel londonien.
        - **Décalage de distribution** : la géographie parisienne, les conventions de prix et la syntaxe française éloignent la requête de la distribution d\u2019entraînement.
        - **Pas de support multilingue** : le modèle a été optimisé uniquement sur des annonces en anglais, le transfert cross-lingue est donc faible et peu fiable.
        - **Ce n\u2019est pas un LLM** : il projette les textes en vecteurs pour la recherche ; il ne raisonne pas, ne traduit pas et ne génère pas d\u2019explications de manière autonome.
        """
    )
