"""Re-exports depuis functions.py pour compatibilite."""
from functions import (
    embeddings_cache_exists,
    load_cached_embeddings_and_dataset,
    compute_corpus_embeddings,
    save_embeddings_to_cache,
    encode_query,
    normalize_rows,
)
