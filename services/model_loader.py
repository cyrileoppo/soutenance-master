from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import prepare_dataset
from utils.constants import GDRIVE_FOLDER_ID, KAGGLE_DATASET, MODEL_CACHE_DIR, ROOT_DIR


# Fichiers requis pour qu'un modèle SentenceTransformer soit valide
_REQUIRED_MODEL_FILES = ['config.json']
_MODEL_WEIGHT_FILES = ['model.safetensors', 'pytorch_model.bin']


def _model_is_valid(model_dir: Path) -> bool:
    """Vérifie que le dossier contient un modèle SentenceTransformer complet."""
    if not model_dir.exists():
        return False
    # config.json doit exister
    if not (model_dir / 'config.json').exists():
        return False
    # Au moins un fichier de poids doit exister
    has_weights = any((model_dir / f).exists() for f in _MODEL_WEIGHT_FILES)
    if not has_weights:
        return False
    # Le fichier de poids ne doit pas être vide ou tronqué (au moins 1MB)
    for f in _MODEL_WEIGHT_FILES:
        weight_path = model_dir / f
        if weight_path.exists() and weight_path.stat().st_size > 1_000_000:
            return True
    return False


def _download_model_from_gdrive(destination: Path) -> None:
    """Télécharge le modèle depuis Google Drive.

    Utilise gdown avec gestion des gros fichiers (confirmation de téléchargement Google).
    """
    import gdown

    destination.mkdir(parents=True, exist_ok=True)

    # Télécharger le dossier complet
    url = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'
    gdown.download_folder(
        url=url,
        output=str(destination),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )

    # Si le fichier safetensors n'a pas été téléchargé correctement (trop gros),
    # essayer de télécharger les fichiers individuellement
    if not _model_is_valid(destination):
        st.warning(
            '\u26a0\ufe0f Le téléchargement automatique du dossier n\'a pas récupéré tous les fichiers. '
            'Tentative de téléchargement individuel des fichiers...'
        )
        # Lister les fichiers du dossier Drive et les télécharger un par un
        try:
            import gdown.download_folder as df_module
            # Fallback: télécharger via l'ID du dossier avec fuzzy
            gdown.download_folder(
                id=GDRIVE_FOLDER_ID,
                output=str(destination),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
            )
        except Exception:
            pass


def _ensure_model_available(model_path: str) -> str:
    """S'assure que le modèle est disponible localement, le télécharge depuis Drive si nécessaire."""
    path = Path(model_path)

    if _model_is_valid(path):
        return str(path)

    # Si le modèle n'est pas valide, tenter le téléchargement
    st.info('\u2b07\ufe0f Téléchargement du modèle fine-tuné depuis Google Drive...')
    download_dest = MODEL_CACHE_DIR
    _download_model_from_gdrive(download_dest)

    if _model_is_valid(download_dest):
        return str(download_dest)

    # Si toujours pas valide, afficher des instructions claires
    missing_info = []
    if not (download_dest / 'config.json').exists():
        missing_info.append('config.json')
    has_any_weight = False
    for f in _MODEL_WEIGHT_FILES:
        fp = download_dest / f
        if fp.exists():
            size_mb = fp.stat().st_size / (1024 * 1024)
            missing_info.append(f'{f} ({size_mb:.1f} MB - possiblement tronqué)')
            has_any_weight = True
        else:
            missing_info.append(f'{f} (manquant)')

    raise FileNotFoundError(
        f'\u274c Le modèle est incomplet dans {download_dest}.\n\n'
        f'Fichiers détectés : {missing_info}\n\n'
        f'\U0001f4cb SOLUTION :\n'
        f'1. Téléchargez manuellement le modèle depuis :\n'
        f'   https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}\n'
        f'2. Placez TOUS les fichiers dans : {download_dest}\n'
        f'3. Ou définissez MODEL_PATH vers un dossier contenant le modèle complet.\n\n'
        f'Fichiers nécessaires : config.json, tokenizer.json, model.safetensors (ou pytorch_model.bin), '
        f'sentence_bert_config.json, modules.json'
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
    """Télécharge le dataset depuis Kaggle via l'API officielle."""
    download_dir = ROOT_DIR / 'data' / 'kaggle_download'
    download_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si déjà téléchargé
    existing = _find_csv(str(download_dir))
    if existing:
        return existing

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)

    csv_path = _find_csv(str(download_dir))
    if not csv_path:
        raise FileNotFoundError(
            f'Aucun fichier CSV trouvé dans {download_dir} après téléchargement du dataset Kaggle ({KAGGLE_DATASET}). '
            f'Vérifiez vos identifiants Kaggle (~/.kaggle/kaggle.json).'
        )
    return csv_path


@st.cache_resource(show_spinner='Chargement du dataset depuis Kaggle...')
def load_dataset() -> pd.DataFrame:
    """Charge le dataset réel depuis Kaggle (kanchana1990/real-estate-data-london-2024)."""
    csv_path = _download_kaggle_dataset()
    df = pd.read_csv(csv_path)
    return prepare_dataset(df)
