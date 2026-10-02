"""Popularity prior, random baseline and the hybrid blending logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recommender.config import POPULARITY_QUANTILE, RANDOM_STATE
from recommender.preprocessing import Index


class PopularityModel:
    """IMDB-style weighted rating: shrinks small-sample averages to the mean."""

    signal = "binary"

    def fit(self, train: pd.DataFrame, index: Index) -> "PopularityModel":
        stats = train.groupby("movie_id")["rating"].agg(["count", "mean"])
        stats = stats.reindex(index.movie_ids)
        counts = stats["count"].fillna(0).to_numpy()
        means = stats["mean"].fillna(0).to_numpy()
        overall = train["rating"].mean()
        m = np.quantile(counts[counts > 0], POPULARITY_QUANTILE)
        weight = counts / (counts + m)
        self.item_counts = counts
        self.scores = weight * means + (1 - weight) * overall
        return self

    def score(self, rows: csr_matrix) -> np.ndarray:
        return np.tile(self.scores, (rows.shape[0], 1))


class RandomModel:
    """Sanity-check baseline: any real model must beat this."""

    signal = "binary"

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def fit(self, *_args, **_kwargs) -> "RandomModel":
        return self

    def score(self, rows: csr_matrix) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.random(rows.shape)


def minmax_rows(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Scale each row to [0, 1] using only candidate (unseen) items.

    Non-candidates become 0. A row with no spread carries no information
    and becomes all zeros.
    """
    lo = np.where(candidates, scores, np.inf).min(axis=1, keepdims=True)
    hi = np.where(candidates, scores, -np.inf).max(axis=1, keepdims=True)
    span = hi - lo
    ok = np.isfinite(span) & (span > 0)
    scaled = (scores - np.where(np.isfinite(lo), lo, 0.0)) / np.where(ok, span, 1.0)
    return np.where(candidates & ok, scaled, 0.0)


def normalize_components(
    components: dict[str, np.ndarray], seen: np.ndarray
) -> dict[str, np.ndarray]:
    """Min-max every component so different score scales can be mixed."""
    candidates = ~seen
    return {name: minmax_rows(s, candidates) for name, s in components.items()}


def combine(
    normalized: dict[str, np.ndarray], weights: dict[str, float], seen: np.ndarray
) -> np.ndarray:
    """Weighted sum of normalised components. Seen items get -inf."""
    total = np.zeros_like(next(iter(normalized.values())))
    for name, weight in weights.items():
        if weight:
            total += weight * normalized[name]
    total[seen] = -np.inf
    return total


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)
