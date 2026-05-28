from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import prepare_dataset
from utils.constants import GDRIVE_FOLDER_ID, KAGGLE_DATASET, MODEL_CACHE_DIR


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
        f'Modèle introuvable à {model_path} et le téléchargement depuis Google Drive a échoué. '
        f'Vérifiez que le dossier Drive est partagé publiquement ou fournissez un MODEL_PATH valide.'
    )


@st.cache_resource(show_spinner='Chargement du modèle fine-tuné...')
def load_model(model_path: str) -> SentenceTransformer:
    """Charge le modèle SentenceTransformer, le télécharge depuis Drive si nécessaire."""
    resolved_path = _ensure_model_available(model_path)
    return SentenceTransformer(resolved_path)


@st.cache_resource(show_spinner='Chargement du dataset depuis Kaggle...')
def load_dataset() -> pd.DataFrame:
    """Charge le dataset réel depuis Kaggle (kanchana1990/real-estate-data-london-2024)."""
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(
            f'Aucun fichier CSV trouvé après téléchargement du dataset Kaggle ({KAGGLE_DATASET}). '
            f'Vérifiez vos identifiants Kaggle et votre connexion internet.'
        )
    df = pd.read_csv(csv_files[0])
    return prepare_dataset(df)
