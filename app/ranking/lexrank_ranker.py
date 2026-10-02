import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity
from app.config import TEXTRANK_DAMPING, SIMILARITY_THRESHOLD


def lexrank_score(matrix: spmatrix) -> list[float]:
    sim = cosine_similarity(matrix)
    n = sim.shape[0]
    # Build stochastic matrix (threshold + row-normalize)
    adj = (sim >= SIMILARITY_THRESHOLD).astype(float)
    np.fill_diagonal(adj, 0)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = adj / row_sums
    # Power iteration
    p = np.ones(n) / n
    for _ in range(100):
        p_new = TEXTRANK_DAMPING * T.T @ p + (1 - TEXTRANK_DAMPING) / n
        if np.linalg.norm(p_new - p) < 1e-6:
            break
        p = p_new
    return p.tolist()