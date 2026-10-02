import networkx as nx
import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity
from app.config import TEXTRANK_DAMPING, SIMILARITY_THRESHOLD


def textrank_score(matrix) -> list:
    sim = cosine_similarity(matrix)
    n = sim.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and sim[i, j] >= SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=float(sim[i, j]))
    if G.number_of_edges() == 0:
        return [1.0 / n] * n
    scores = nx.pagerank(G, alpha=TEXTRANK_DAMPING, max_iter=100)
    return [scores.get(i, 0.0) for i in range(n)]