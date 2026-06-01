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
    """Télécharge le dossier du modèle fine-tuné depuis Google Drive via gdown."""
    import gdown

    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'
    gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False)


def _ensure_model_available(model_path: str) -> str:
    """S'assure que le modèle est disponible localement, le télécharge depuis Drive si nécessaire."""
    path = Path(model_path)

    if path.exists() and any(path.iterdir()):
        return str(path)

    st.info('\u2b07\ufe0f Téléchargement du modèle fine-tuné depuis Google Drive (premier lancement uniquement)...')
    download_dest = MODEL_CACHE_DIR
    _download_model_from_gdrive(download_dest)

    if download_dest.exists() and any(download_dest.iterdir()):
        return str(download_dest)

    raise FileNotFoundError(
        f'Modèle introuvable \u00e0 {model_path} et le téléchargement depuis Google Drive a échoué. '
        f'Vérifiez que le dossier Drive est partagé publiquement ou fournissez un MODEL_PATH valide.'
    )


@st.cache_resource(show_spinner='Chargement du modèle fine-tuné...')
def load_model(model_path: str) -> SentenceTransformer:
    """Charge le modèle SentenceTransformer, le télécharge depuis Drive si nécessaire."""
    resolved_path = _ensure_model_available(model_path)
    return SentenceTransformer(resolved_path)


def _find_csv(directory: str) -> str | None:
    """Cherche un fichier CSV dans un dossier (récursivement)."""
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if csv_files:
        return csv_files[0]
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    if csv_files:
        return csv_files[0]
    return None


def _download_kaggle_dataset() -> str:
    """Télécharge le dataset depuis Kaggle via l'API officielle `kaggle`.

    Prérequis : pip install kaggle + fichier ~/.kaggle/kaggle.json
    Le dataset est téléchargé une seule fois et mis en cache dans data/kaggle_download/.
    """
    download_dir = ROOT_DIR / 'data' / 'kaggle_download'
    download_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si déjà téléchargé
    existing = _find_csv(str(download_dir))
    if existing:
        return existing

    # Télécharger via l'API Kaggle officielle
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)

    # Trouver le CSV téléchargé
    csv_path = _find_csv(str(download_dir))
    if not csv_path:
        raise FileNotFoundError(
            f'Aucun fichier CSV trouvé dans {download_dir} après téléchargement du dataset Kaggle ({KAGGLE_DATASET}). '
            f'Vérifiez vos identifiants Kaggle (~/.kaggle/kaggle.json).'
        )
    return csv_path


@st.cache_resource(show_spinner='Chargement du dataset depuis Kaggle...')
def load_dataset() -> pd.DataFrame:
    """Charge le dataset réel depuis Kaggle (kanchana1990/real-estate-data-london-2024).

    Utilise l'API Kaggle officielle (package `kaggle`).
    Le dataset est téléchargé et dézippé automatiquement au premier lancement.
    """
    csv_path = _download_kaggle_dataset()
    df = pd.read_csv(csv_path)
    return prepare_dataset(df)
