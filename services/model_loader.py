from __future__ import annotations

import glob
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import prepare_dataset
from utils.constants import GDRIVE_FOLDER_ID, KAGGLE_DATASET, MODEL_CACHE_DIR, ROOT_DIR


def _download_model_from_gdrive(destination: Path) -> None:
    """T\u00e9l\u00e9charge le dossier du mod\u00e8le fine-tun\u00e9 depuis Google Drive via gdown."""
    import gdown

    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'
    gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False)


def _ensure_model_available(model_path: str) -> str:
    """S'assure que le mod\u00e8le est disponible localement, le t\u00e9l\u00e9charge depuis Drive si n\u00e9cessaire."""
    path = Path(model_path)

    if path.exists() and any(path.iterdir()):
        return str(path)

    st.info('\u2b07\ufe0f T\u00e9l\u00e9chargement du mod\u00e8le fine-tun\u00e9 depuis Google Drive (premier lancement uniquement)...')
    download_dest = MODEL_CACHE_DIR
    _download_model_from_gdrive(download_dest)

    if download_dest.exists() and any(download_dest.iterdir()):
        return str(download_dest)

    raise FileNotFoundError(
        f'Mod\u00e8le introuvable \u00e0 {model_path} et le t\u00e9l\u00e9chargement depuis Google Drive a \u00e9chou\u00e9. '
        f'V\u00e9rifiez que le dossier Drive est partag\u00e9 publiquement ou fournissez un MODEL_PATH valide.'
    )


@st.cache_resource(show_spinner='Chargement du mod\u00e8le fine-tun\u00e9...')
def load_model(model_path: str) -> SentenceTransformer:
    """Charge le mod\u00e8le SentenceTransformer, le t\u00e9l\u00e9charge depuis Drive si n\u00e9cessaire."""
    resolved_path = _ensure_model_available(model_path)
    return SentenceTransformer(resolved_path)


def _find_csv(directory: str) -> str | None:
    """Cherche un fichier CSV dans un dossier (r\u00e9cursivement)."""
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if csv_files:
        return csv_files[0]
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    if csv_files:
        return csv_files[0]
    return None


def _download_kaggle_dataset() -> str:
    """T\u00e9l\u00e9charge le dataset depuis Kaggle via l'API officielle.

    Pr\u00e9requis : pip install kaggle + fichier ~/.kaggle/kaggle.json
    Le dataset est t\u00e9l\u00e9charg\u00e9 une seule fois et mis en cache dans data/kaggle_download/.
    """
    download_dir = ROOT_DIR / 'data' / 'kaggle_download'
    download_dir.mkdir(parents=True, exist_ok=True)

    # V\u00e9rifier si d\u00e9j\u00e0 t\u00e9l\u00e9charg\u00e9
    existing = _find_csv(str(download_dir))
    if existing:
        return existing

    # V\u00e9rifier la pr\u00e9sence des identifiants Kaggle avant de tenter le t\u00e9l\u00e9chargement
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    kaggle_env = os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
    if not kaggle_json.exists() and not kaggle_env:
        raise FileNotFoundError(
            'Identifiants Kaggle introuvables. Pour t\u00e9l\u00e9charger le dataset automatiquement, '
            'cr\u00e9ez le fichier ~/.kaggle/kaggle.json ou d\u00e9finissez les variables d\u2019environnement '
            'KAGGLE_USERNAME et KAGGLE_KEY. '
            'Alternativement, placez le cache Parquet dans data/dataset_with_embeddings.parquet.'
        )

    # T\u00e9l\u00e9charger via l'API Kaggle officielle
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)

    # Trouver le CSV t\u00e9l\u00e9charg\u00e9
    csv_path = _find_csv(str(download_dir))
    if not csv_path:
        raise FileNotFoundError(
            f'Aucun fichier CSV trouv\u00e9 dans {download_dir} apr\u00e8s t\u00e9l\u00e9chargement du dataset Kaggle ({KAGGLE_DATASET}). '
            f'V\u00e9rifiez vos identifiants Kaggle (~/.kaggle/kaggle.json).'
        )
    return csv_path


@st.cache_resource(show_spinner='Chargement du dataset depuis Kaggle...')
def load_dataset() -> pd.DataFrame:
    """Charge le dataset r\u00e9el depuis Kaggle (kanchana1990/real-estate-data-london-2024).

    Utilise l'API Kaggle officielle (package `kaggle`).
    Le dataset est t\u00e9l\u00e9charg\u00e9 et d\u00e9zipp\u00e9 automatiquement au premier lancement.
    """
    csv_path = _download_kaggle_dataset()
    df = pd.read_csv(csv_path)
    return prepare_dataset(df)
