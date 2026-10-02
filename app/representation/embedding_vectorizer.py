"""Optional sentence embedding hook — disabled by default."""
from typing import List
import numpy as np

_model = None


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Returns sentence embeddings using sentence-transformers if available."""
    global _model
    try:
        from sentence_transformers import SentenceTransformer
        if _model is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model.encode(texts, show_progress_bar=False)
    except ImportError:
        raise RuntimeError("sentence-transformers not installed; use TF-IDF mode")