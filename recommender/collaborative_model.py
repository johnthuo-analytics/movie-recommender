"""Collaborative-filtering models built from user-item interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

from recommender.config import (
    KNN_NEIGHBORS,
    KNN_SHRINKAGE,
    RANDOM_STATE,
    SVD_FACTORS,
)
from recommender.preprocessing import Index, build_matrix


def prune_top_k(similarity: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest values in each row; zero out the rest."""
    if k >= similarity.shape[1]:
        return similarity
    keep = np.argpartition(similarity, -k, axis=1)[:, -k:]
    pruned = np.zeros_like(similarity)
    rows = np.arange(similarity.shape[0])[:, None]
    pruned[rows, keep] = similarity[rows, keep]
    return pruned


class ItemKNN:
    """Item-item collaborative filtering.

    Similarity is the cosine between item columns of the chosen signal
    ('centered' gives adjusted cosine), damped by a shrinkage term so that
    pairs with few common raters are not trusted, then pruned to the top
    neighbours per item.
    """

    def __init__(
        self,
        signal: str = "centered",
        n_neighbors: int = KNN_NEIGHBORS,
        shrinkage: float = KNN_SHRINKAGE,
    ):
        self.signal = signal
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage

    def fit(self, train: pd.DataFrame, index: Index) -> "ItemKNN":
        matrix = build_matrix(train, index, self.signal)
        binary = build_matrix(train, index, "binary")
        items = normalize(matrix.T.tocsr())  # one unit vector per item
        similarity = (items @ items.T).toarray()
        common = (binary.T @ binary).toarray()  # users who rated both items
        similarity *= common / (common + self.shrinkage)
        np.fill_diagonal(similarity, 0.0)
        similarity = np.maximum(similarity, 0.0)
        self.similarity = csr_matrix(prune_top_k(similarity, self.n_neighbors))
        return self

    def score(self, rows: csr_matrix) -> np.ndarray:
        """rows: (n_users x n_items) matrix built with self.signal."""
        return (rows @ self.similarity.T).toarray()

    def similar_items(self, item_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.similarity[item_idx].toarray().ravel()
        top = np.argsort(-row, kind="stable")[:k]
        top = top[row[top] > 0]  # drop items with no neighbour signal
        return top, row[top]


class SVDModel:
    """PureSVD: truncated SVD of the user-item matrix.

    Scores are the low-rank reconstruction X V V^T. Folding a user in only
    needs their history row, so users outside the training matrix also work.
    """

    def __init__(
        self,
        signal: str = "centered",
        n_factors: int = SVD_FACTORS,
        random_state: int = RANDOM_STATE,
    ):
        self.signal = signal
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, train: pd.DataFrame, index: Index) -> "SVDModel":
        matrix = build_matrix(train, index, self.signal)
        k = max(1, min(self.n_factors, min(matrix.shape) - 1))
        v0 = np.random.default_rng(self.random_state).random(min(matrix.shape))
        _, _, vt = svds(matrix, k=k, v0=v0)
        self.item_factors = vt.T  # (n_items x k)
        return self

    def score(self, rows: csr_matrix) -> np.ndarray:
        return (rows @ self.item_factors) @ self.item_factors.T
