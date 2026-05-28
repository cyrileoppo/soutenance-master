from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import prepare_dataset


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_path)


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
        import kagglehub  # noqa: F401

        path = kagglehub.dataset_download("kanchana1990/real-estate-data-london-2024")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])
    except Exception:
        pass
    return None
