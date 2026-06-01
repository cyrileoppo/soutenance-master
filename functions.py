"""functions.py - Fichier centralisant TOUTES les fonctions metier du projet.

Toute nouvelle fonction doit etre ajoutee ici.
Les autres fichiers (services/, utils/, views/) importent depuis ce module.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi


# =============================================================================
# CONSTANTES
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

GDRIVE_FOLDER_ID = '1-VAMOQmhsLkTbgsgff0-iEnSaPrjCTcY'
GDRIVE_FOLDER_URL = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'
MODEL_CACHE_DIR = ROOT_DIR / 'model' / 'final_model'
MODEL_PATH = os.getenv('MODEL_PATH', str(MODEL_CACHE_DIR))
KAGGLE_DATASET = 'kanchana1990/real-estate-data-london-2024'
ASSETS_PATH = ROOT_DIR / 'assets' / 'styles.css'
EMBEDDINGS_CACHE_PATH = ROOT_DIR / 'data' / 'dataset_with_embeddings.parquet'

TOP_K = 3
RANDOM_STATE = 42
EMBEDDING_PROJECTION_NEIGHBORS = 10
SPECIAL_TOKENS = ('[TITLE]', '[DESC]', '[ATTR]')
PIPELINE_STAGES = (
    'Encodage de l\u2019annonce...',
    'G\u00e9n\u00e9ration de l\u2019embedding...',
    'Calcul des similarit\u00e9s s\u00e9mantiques...',
    'Classement des voisins les plus proches...',
)
COLOR_OPTIONS = {
    'Type de bien': 'propertyType',
    'Chambres': 'bedrooms',
    'Gamme de prix': 'price_range',
}
LIMITATION_QUERIES = [
    {
        'label': 'Requ\u00eate en fran\u00e7ais',
        'query_text': '[TITLE] Appartement lumineux \u00e0 Paris [ATTR] Type: Apartment [ATTR] Beds: 2 [ATTR] Baths: 1 [ATTR] Size: 700 sqft [ATTR] Price: \u20ac850,000',
        'explanation': 'L\u2019encodeur a \u00e9t\u00e9 fine-tun\u00e9 uniquement sur des annonces londoniennes en anglais.',
    },
    {
        'label': 'Annonce parisienne',
        'query_text': '[TITLE] Haussmann apartment near Parc Monceau [ATTR] Type: Apartment [ATTR] Beds: 3 [ATTR] Baths: 2 [ATTR] Size: 1450 sqft [ATTR] Price: \u20ac1,950,000',
        'explanation': 'G\u00e9ographie et conventions de prix parisiennes hors distribution d\u2019entra\u00eenement.',
    },
    {
        'label': 'Texte hors immobilier',
        'query_text': '[DESC] Transformer models improve retrieval by mapping semantically related sentences close together in vector space.',
        'explanation': 'Le mod\u00e8le n\u2019est pas un encodeur g\u00e9n\u00e9raliste. Texte hors domaine = correspondances faibles.',
    },
]

COLUMNS_TO_FILL = ['title', 'propertyType', 'sizeSqFeetMax', 'bedrooms', 'bathrooms', 'price']


# =============================================================================
# HELPERS
# =============================================================================


def format_currency(value: str) -> str:
    return value if isinstance(value, str) and value else 'N/A'


def parse_price(value: str) -> int:
    digits = re.sub(r'[^0-9]', '', str(value))
    return int(digits) if digits else 0


def price_range_label(value: str) -> str:
    amount = parse_price(value)
    if amount < 600_000:
        return 'Moins de \u00a3600k'
    if amount < 1_000_000:
        return '\u00a3600k - \u00a3999k'
    if amount < 1_500_000:
        return '\u00a31.0m - \u00a31.49m'
    if amount < 2_000_000:
        return '\u00a31.5m - \u00a31.99m'
    return '\u00a32.0m+'


def similarity_verdict(score: float) -> str:
    if score >= 0.85:
        return 'Tr\u00e8s similaire'
    if score >= 0.65:
        return 'Mod\u00e9r\u00e9ment similaire'
    return 'Non reli\u00e9'


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


# =============================================================================
# PRETRAITEMENT
# =============================================================================


def clean_html(html_text: Any) -> str:
    if not isinstance(html_text, str):
        return ''
    return BeautifulSoup(html_text, 'html.parser').get_text(separator=' ').strip()


def format_decimal_string(value: str) -> str:
    return value.replace('.0', '') if value.endswith('.0') else value


def build_rich_anchor(row: pd.Series) -> str:
    return (
        row['title']
        + ' [ATTR] Type: ' + row['propertyType']
        + ' [ATTR] Beds: ' + row['bedrooms']
        + ' [ATTR] Baths: ' + row['bathrooms']
        + ' [ATTR] Size: ' + row['sizeSqFeetMax'] + ' sqft'
        + ' [ATTR] Price: ' + row['price']
    )


def encode_anchor_text(anchor: str) -> str:
    return '[TITLE] ' + anchor


def encode_description_text(description: str) -> str:
    return '[DESC] ' + description


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared['clean_description'] = prepared['descriptionHtml'].apply(clean_html)

    for column in COLUMNS_TO_FILL:
        prepared[column] = prepared[column].fillna('N/A').astype(str)

    prepared['bedrooms'] = prepared['bedrooms'].apply(format_decimal_string)
    prepared['bathrooms'] = prepared['bathrooms'].apply(format_decimal_string)
    prepared['rich_anchor'] = prepared.apply(build_rich_anchor, axis=1)
    prepared['anchor_text'] = prepared['rich_anchor'].apply(encode_anchor_text)
    prepared['description_text'] = prepared['clean_description'].apply(encode_description_text)
    prepared['price_range'] = prepared['price'].apply(price_range_label)
    prepared = prepared.reset_index(drop=True)
    prepared['listing_id'] = prepared.index
    return prepared


# =============================================================================
# CHARGEMENT MODELE & DATASET
# =============================================================================


def find_csv(directory: str) -> str | None:
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if csv_files:
        return csv_files[0]
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    if csv_files:
        return csv_files[0]
    return None


def download_model_from_gdrive(destination: Path) -> None:
    import gdown
    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'
    gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False)


def ensure_model_available(model_path: str) -> str:
    path = Path(model_path)
    if path.exists() and any(path.iterdir()):
        return str(path)

    download_dest = MODEL_CACHE_DIR
    download_model_from_gdrive(download_dest)

    if download_dest.exists() and any(download_dest.iterdir()):
        return str(download_dest)

    raise FileNotFoundError(
        f'Modele introuvable a {model_path} et le telechargement depuis Google Drive a echoue.'
    )


def download_kaggle_dataset() -> str:
    download_dir = ROOT_DIR / 'data' / 'kaggle_download'
    download_dir.mkdir(parents=True, exist_ok=True)

    existing = find_csv(str(download_dir))
    if existing:
        return existing

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)

    csv_path = find_csv(str(download_dir))
    if not csv_path:
        raise FileNotFoundError(
            f'Aucun fichier CSV trouve dans {download_dir} apres telechargement.'
        )
    return csv_path


# =============================================================================
# EMBEDDINGS
# =============================================================================


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embeddings_cache_exists() -> bool:
    if not EMBEDDINGS_CACHE_PATH.exists():
        return False
    try:
        df = pd.read_parquet(EMBEDDINGS_CACHE_PATH, columns=['anchor_embedding'])
        return len(df) > 0
    except Exception:
        return False


def load_cached_embeddings_and_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(EMBEDDINGS_CACHE_PATH)

    anchor_embeddings = np.array(df['anchor_embedding'].tolist(), dtype=float)
    description_embeddings = np.array(df['description_embedding'].tolist(), dtype=float)
    profile_embeddings = np.array(df['profile_embedding'].tolist(), dtype=float)

    dataset = df.drop(columns=['anchor_embedding', 'description_embedding', 'profile_embedding'])

    embedding_bundle = {
        'anchor_embeddings': anchor_embeddings,
        'description_embeddings': description_embeddings,
        'profile_embeddings': profile_embeddings,
    }
    return dataset, embedding_bundle


def compute_corpus_embeddings(model, dataset: pd.DataFrame) -> dict[str, Any]:
    anchor_embeddings = model.encode(dataset['anchor_text'].tolist(), normalize_embeddings=True)
    description_embeddings = model.encode(dataset['description_text'].tolist(), normalize_embeddings=True)
    profile_embeddings = normalize_rows((anchor_embeddings + description_embeddings) / 2.0)
    return {
        'anchor_embeddings': np.asarray(anchor_embeddings, dtype=float),
        'description_embeddings': np.asarray(description_embeddings, dtype=float),
        'profile_embeddings': np.asarray(profile_embeddings, dtype=float),
    }


def save_embeddings_to_cache(dataset: pd.DataFrame, embedding_bundle: dict[str, Any]) -> None:
    df_to_save = dataset.copy()
    df_to_save['anchor_embedding'] = list(embedding_bundle['anchor_embeddings'])
    df_to_save['description_embedding'] = list(embedding_bundle['description_embeddings'])
    df_to_save['profile_embedding'] = list(embedding_bundle['profile_embeddings'])

    EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_parquet(EMBEDDINGS_CACHE_PATH, index=False)


def encode_query(model, text: str) -> np.ndarray:
    embedding = model.encode([text], normalize_embeddings=True)
    return np.asarray(embedding[0], dtype=float)


# =============================================================================
# SIMILARITE & RECHERCHE
# =============================================================================


def cosine_scores(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    return corpus_embeddings @ query_embedding


def cosine_similarity(left_embedding: np.ndarray, right_embedding: np.ndarray) -> float:
    return float(np.dot(left_embedding, right_embedding))


def vector_distance(score: float) -> float:
    return 1.0 - score


def embedding_position(dataset: pd.DataFrame, listing_id: int) -> int:
    matches = dataset.index[dataset['listing_id'] == listing_id]
    if matches.empty:
        raise KeyError(f'listing_id inconnu : {listing_id}')
    return int(matches[0])


def semantic_search(
    model,
    dataset: pd.DataFrame,
    description_embeddings: np.ndarray,
    query_text: str,
    top_k: int,
    exclude_listing_id: int | None = None,
) -> pd.DataFrame:
    query_embedding = encode_query(model, query_text)
    scores = cosine_scores(query_embedding, description_embeddings)
    ranked = dataset.copy()
    ranked['similarity'] = scores
    if exclude_listing_id is not None:
        ranked = ranked[ranked['listing_id'] != exclude_listing_id]
    return ranked.sort_values('similarity', ascending=False).head(top_k).reset_index(drop=True)


def nearest_neighbors(
    dataset: pd.DataFrame,
    profile_embeddings: np.ndarray,
    listing_id: int,
    top_k: int,
) -> pd.DataFrame:
    query_embedding = profile_embeddings[embedding_position(dataset, listing_id)]
    scores = cosine_scores(query_embedding, profile_embeddings)
    ranked = dataset.copy()
    ranked['similarity'] = scores
    ranked = ranked[ranked['listing_id'] != listing_id]
    return ranked.sort_values('similarity', ascending=False).head(top_k).reset_index(drop=True)


def compare_listings(
    anchor_embeddings: np.ndarray,
    description_embeddings: np.ndarray,
    dataset: pd.DataFrame,
    left_listing_id: int,
    right_listing_id: int,
) -> float:
    left_pos = embedding_position(dataset, left_listing_id)
    right_pos = embedding_position(dataset, right_listing_id)
    left_to_right = cosine_similarity(anchor_embeddings[left_pos], description_embeddings[right_pos])
    right_to_left = cosine_similarity(anchor_embeddings[right_pos], description_embeddings[left_pos])
    return float((left_to_right + right_to_left) / 2.0)


# =============================================================================
# BM25
# =============================================================================


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(dataset: pd.DataFrame) -> dict[str, Any]:
    corpus = dataset['clean_description'].tolist()
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    return {'bm25': BM25Okapi(tokenized_corpus), 'corpus': corpus}


def bm25_search(
    bm25_index: dict[str, Any],
    dataset: pd.DataFrame,
    query_text: str,
    top_k: int,
    exclude_listing_id: int | None = None,
) -> pd.DataFrame:
    scores = bm25_index['bm25'].get_scores(tokenize(query_text))
    ranked = dataset.copy()
    ranked['bm25_score'] = scores
    if exclude_listing_id is not None:
        ranked = ranked[ranked['listing_id'] != exclude_listing_id]
    return ranked.sort_values('bm25_score', ascending=False).head(top_k).reset_index(drop=True)


# =============================================================================
# VISUALISATION
# =============================================================================


def compute_projection(dataset: pd.DataFrame, profile_embeddings, color_by: str) -> pd.DataFrame:
    import umap
    reducer = umap.UMAP(n_neighbors=EMBEDDING_PROJECTION_NEIGHBORS, min_dist=0.15, random_state=RANDOM_STATE)
    coords = reducer.fit_transform(profile_embeddings)
    projection = dataset.copy()
    projection['umap_x'] = coords[:, 0]
    projection['umap_y'] = coords[:, 1]
    projection['color_value'] = projection[color_by].astype(str)
    return projection


def build_embedding_figure(projection: pd.DataFrame):
    import plotly.express as px
    fig = px.scatter(
        projection,
        x='umap_x',
        y='umap_y',
        color='color_value',
        hover_name='title',
        hover_data={
            'propertyType': True,
            'bedrooms': True,
            'bathrooms': True,
            'price': True,
            'umap_x': False,
            'umap_y': False,
            'color_value': False,
        },
        custom_data=['listing_id'],
        template='plotly_dark',
        opacity=0.88,
        height=620,
    )
    fig.update_traces(marker={'size': 14, 'line': {'width': 0}}, selector={'mode': 'markers'})
    fig.update_layout(
        margin={'l': 0, 'r': 0, 't': 16, 'b': 0},
        legend_title_text='',
        xaxis_title='Dimension UMAP 1',
        yaxis_title='Dimension UMAP 2',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.35)',
    )
    return fig
