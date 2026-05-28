from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import prepare_dataset
from utils.constants import GDRIVE_FOLDER_ID, MODEL_CACHE_DIR


def _download_model_from_gdrive(destination: Path) -> None:
    """Download the fine-tuned model folder from Google Drive using gdown.

    The folder must be shared publicly (anyone with the link).
    Downloads all files in the folder to the destination directory.
    """
    import gdown

    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'
    gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False)


def _ensure_model_available(model_path: str) -> str:
    """Ensure the model is available locally, downloading from Drive if needed.

    Returns the path to use for SentenceTransformer loading.
    """
    path = Path(model_path)

    # Check if the model is already available locally
    if path.exists() and any(path.iterdir()):
        return str(path)

    # If MODEL_PATH points to a non-existent location, try downloading
    st.info('⬇️ Downloading fine-tuned model from Google Drive (first run only)...')
    download_dest = MODEL_CACHE_DIR
    _download_model_from_gdrive(download_dest)

    if download_dest.exists() and any(download_dest.iterdir()):
        return str(download_dest)

    raise FileNotFoundError(
        f'Model not found at {model_path} and download from Google Drive failed. '
        f'Please ensure the Drive folder is publicly shared or provide a valid MODEL_PATH.'
    )


@st.cache_resource(show_spinner='Loading fine-tuned model...')
def load_model(model_path: str) -> SentenceTransformer:
    """Load the SentenceTransformer model, downloading from Drive if necessary."""
    resolved_path = _ensure_model_available(model_path)
    return SentenceTransformer(resolved_path)


@st.cache_resource(show_spinner=False)
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset from Kaggle (preferred) or fall back to local CSV.

    Tries to download the real London real estate dataset from Kaggle using
    kagglehub. If that fails (no credentials, no network, etc.), falls back
    to the local sample CSV file.
    """
    df = _try_load_from_kaggle()
    if df is None:
        df = pd.read_csv(Path(csv_path))
    return prepare_dataset(df)


def _try_load_from_kaggle() -> pd.DataFrame | None:
    """Attempt to download and load the real dataset from Kaggle."""
    try:
        import kagglehub

        path = kagglehub.dataset_download('kanchana1990/real-estate-data-london-2024')
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        if csv_files:
            return pd.read_csv(csv_files[0])
    except Exception:
        pass
    return None
