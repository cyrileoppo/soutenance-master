from __future__ import annotations

import glob
import os
import zipfile
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
        f'Modèle introuvable \u00e0 {model_path} et le téléchargement depuis Google Drive a échoué. '
        f'Vérifiez que le dossier Drive est partagé publiquement ou fournissez un MODEL_PATH valide.'
    )


@st.cache_resource(show_spinner='Chargement du modèle fine-tuné...')
def load_model(model_path: str) -> SentenceTransformer:
    """Charge le modèle SentenceTransformer, le télécharge depuis Drive si nécessaire."""
    resolved_path = _ensure_model_available(model_path)
    return SentenceTransformer(resolved_path)


def _download_kaggle_dataset() -> str:
    """Télécharge le dataset depuis Kaggle via l'API officielle.

    Utilise la commande kaggle CLI en fallback si l'API Python ne fonctionne pas.
    Nécessite un fichier ~/.kaggle/kaggle.json avec vos identifiants.
    """
    from utils.constants import ROOT_DIR

    download_dir = ROOT_DIR / 'data' / 'kaggle_download'
    download_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si déjà téléchargé
    existing_csvs = glob.glob(os.path.join(str(download_dir), '*.csv'))
    if existing_csvs:
        return existing_csvs[0]

    try:
        # Méthode 1 : API Kaggle Python
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)
    except Exception:
        # Méthode 2 : opendatasets
        try:
            import opendatasets as od
            od.download(f'https://www.kaggle.com/datasets/{KAGGLE_DATASET}', data_dir=str(download_dir))
        except Exception:
            # Méthode 3 : kaggle CLI
            os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p {download_dir} --unzip')

    # Chercher le CSV (peut être dans un sous-dossier)
    csv_files = glob.glob(os.path.join(str(download_dir), '**', '*.csv'), recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f'Aucun fichier CSV trouvé après téléchargement du dataset Kaggle ({KAGGLE_DATASET}). '
            f'Vérifiez vos identifiants Kaggle (~/.kaggle/kaggle.json) et votre connexion internet.'
        )
    return csv_files[0]


@st.cache_resource(show_spinner='Chargement du dataset depuis Kaggle...')
def load_dataset() -> pd.DataFrame:
    """Charge le dataset réel depuis Kaggle (kanchana1990/real-estate-data-london-2024)."""
    csv_path = _download_kaggle_dataset()
    df = pd.read_csv(csv_path)
    return prepare_dataset(df)
