from __future__ import annotations

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
    dataset = pd.read_csv(Path(csv_path))
    return prepare_dataset(dataset)
