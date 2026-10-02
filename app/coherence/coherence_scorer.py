"""Score local coherence of a section as mean cosine similarity of adjacent pairs."""
from typing import List

import numpy as np

from app.metadata.schema import SentenceMeta


def score_adjacent_pairs(sentences: List[SentenceMeta], matrix) -> float:
    """
    Compute mean cosine similarity of adjacent sentence pairs.
    Also writes the score onto each sentence's coherence_score field.
    Returns the mean score (0.0 if unavailable).
    """
    if len(sentences) < 2 or matrix is None:
        return 0.0

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        arr = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        scores = []
        for i in range(len(sentences) - 1):
            a = arr[i].reshape(1, -1)
            b = arr[i + 1].reshape(1, -1)
            scores.append(float(cosine_similarity(a, b)[0][0]))

        mean_coh = float(np.mean(scores))
        for s in sentences:
            s.coherence_score = mean_coh
        return mean_coh

    except Exception:
        return 0.0