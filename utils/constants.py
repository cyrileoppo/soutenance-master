from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Google Drive folder ID pour le modèle fine-tuné
GDRIVE_FOLDER_ID = '1-VAMOQmhsLkTbgsgff0-iEnSaPrjCTcY'
GDRIVE_FOLDER_URL = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'

# Chemin local où le modèle sera mis en cache après téléchargement
MODEL_CACHE_DIR = ROOT_DIR / 'model' / 'final_model'

# Peut être surchargé via variable d'environnement
MODEL_PATH = os.getenv('MODEL_PATH', str(MODEL_CACHE_DIR))

# Dataset Kaggle
KAGGLE_DATASET = 'kanchana1990/real-estate-data-london-2024'

ASSETS_PATH = ROOT_DIR / 'assets' / 'styles.css'
MODEL_README_PATH = ROOT_DIR / 'model' / 'README.md'
TOP_K = 3
RANDOM_STATE = 42
EMBEDDING_PROJECTION_NEIGHBORS = 10
SPECIAL_TOKENS = ('[TITLE]', '[DESC]', '[ATTR]')
PIPELINE_STAGES = (
    'Encodage de l\u2019annonce...',
    'Génération de l\u2019embedding...',
    'Calcul des similarités sémantiques...',
    'Classement des voisins les plus proches...',
)
COLOR_OPTIONS = {
    'Type de bien': 'propertyType',
    'Chambres': 'bedrooms',
    'Gamme de prix': 'price_range',
}
LIMITATION_QUERIES = [
    {
        'label': 'Requête en français',
        'query_text': '[TITLE] Appartement lumineux \u00e0 Paris [ATTR] Type: Apartment [ATTR] Beds: 2 [ATTR] Baths: 1 [ATTR] Size: 700 sqft [ATTR] Price: \u20ac850,000',
        'explanation': 'L\u2019encodeur a été fine-tuné uniquement sur des annonces londoniennes en anglais. Le français et les prix en euros créent un décalage de distribution clair.',
    },
    {
        'label': 'Annonce parisienne',
        'query_text': '[TITLE] Haussmann apartment near Parc Monceau [ATTR] Type: Apartment [ATTR] Beds: 3 [ATTR] Baths: 2 [ATTR] Size: 1450 sqft [ATTR] Price: \u20ac1,950,000',
        'explanation': 'C\u2019est toujours de l\u2019immobilier, mais la géographie et les conventions de prix parisiennes sortent de la distribution d\u2019entraînement londonienne.',
    },
    {
        'label': 'Texte hors immobilier',
        'query_text': '[DESC] Transformer models improve retrieval by mapping semantically related sentences close together in vector space.',
        'explanation': 'Le modèle n\u2019est pas un encodeur de phrases généraliste. Un texte hors domaine produit des correspondances faibles et incohérentes.',
    },
]
