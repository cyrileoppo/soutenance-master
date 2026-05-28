from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Google Drive folder ID for the fine-tuned model
GDRIVE_FOLDER_ID = '1-VAMOQmhsLkTbgsgff0-iEnSaPrjCTcY'
GDRIVE_FOLDER_URL = f'https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}'

# Local path where the model will be cached after download
MODEL_CACHE_DIR = ROOT_DIR / 'model' / 'final_model'

# Override with env var or mounted path if available
MODEL_PATH = os.getenv('MODEL_PATH', str(MODEL_CACHE_DIR))

DATA_PATH = ROOT_DIR / 'data' / 'sample_listings.csv'
ASSETS_PATH = ROOT_DIR / 'assets' / 'styles.css'
MODEL_README_PATH = ROOT_DIR / 'model' / 'README.md'
TOP_K = 3
RANDOM_STATE = 42
EMBEDDING_PROJECTION_NEIGHBORS = 10
SPECIAL_TOKENS = ('[TITLE]', '[DESC]', '[ATTR]')
PIPELINE_STAGES = (
    'Encoding listing...',
    'Generating embedding...',
    'Computing semantic similarities...',
    'Ranking nearest neighbors...',
)
COLOR_OPTIONS = {
    'Property type': 'propertyType',
    'Bedrooms': 'bedrooms',
    'Price range': 'price_range',
}
LIMITATION_QUERIES = [
    {
        'label': 'French query',
        'query_text': '[TITLE] Appartement lumineux \u00e0 Paris [ATTR] Type: Apartment [ATTR] Beds: 2 [ATTR] Baths: 1 [ATTR] Size: 700 sqft [ATTR] Price: \u20ac850,000',
        'explanation': 'The encoder was fine-tuned only on English London listings, so French phrasing and euro pricing create a clear distribution shift.',
    },
    {
        'label': 'Parisian listing',
        'query_text': '[TITLE] Haussmann apartment near Parc Monceau [ATTR] Type: Apartment [ATTR] Beds: 3 [ATTR] Baths: 2 [ATTR] Size: 1450 sqft [ATTR] Price: \u20ac1,950,000',
        'explanation': 'This is still real estate, but it sits outside the London-specific geography and pricing distribution seen during training.',
    },
    {
        'label': 'Non real-estate text',
        'query_text': '[DESC] Transformer models improve retrieval by mapping semantically related sentences close together in vector space.',
        'explanation': 'The model is not a general-purpose sentence encoder, so off-domain language should yield weak and incoherent matches.',
    },
]
