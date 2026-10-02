"""Content-based model: movies are similar when their genres are similar."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from recommender.config import GENRE_COLS
from recommender.preprocessing import Index


class ContentModel:
    """Cosine similarity over binary genre vectors.

    The 19 genre flags are already numeric, so no TF-IDF step is needed.
    A user's score for a movie is its mean similarity to the movies they liked.
    """

    signal = "liked"

    def fit(self, movies: pd.DataFrame, index: Index) -> "ContentModel":
        genres = movies.set_index("movie_id").loc[index.movie_ids, GENRE_COLS].to_numpy(float)
        self.genre_matrix = genres
        self.similarity = cosine_similarity(genres)
        return self

    def score(self, rows: csr_matrix) -> np.ndarray:
        """rows: (n_users x n_items) 'liked' matrix. Returns (n_users x n_items)."""
        n_liked = np.asarray(rows.sum(axis=1)).ravel()
        totals = np.asarray(rows @ self.similarity)
        return totals / np.maximum(n_liked, 1)[:, None]

    def similar_items(self, item_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k most similar item positions (excluding the item) and scores."""
        scores = self.similarity[item_idx].copy()
        scores[item_idx] = -np.inf
        top = np.argsort(-scores, kind="stable")[:k]
        return top, scores[top]
