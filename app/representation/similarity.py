from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix
import numpy as np


def pairwise_cosine(matrix: spmatrix) -> np.ndarray:
    return cosine_similarity(matrix)


def top_k_similar(query_vec, matrix: spmatrix, k: int = 5):
    sims = cosine_similarity(query_vec, matrix).flatten()
    return sims.argsort()[::-1][:k]