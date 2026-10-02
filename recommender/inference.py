"""The trained recommender: one object that the API loads and serves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from recommender.collaborative_model import ItemKNN, SVDModel
from recommender.config import (
    BEST_CONFIG_PATH,
    COLD_START_GENRE_BOOST,
    COMPONENTS,
    DEFAULT_KNN_SIGNAL,
    DEFAULT_SVD_SIGNAL,
    DEFAULT_WEIGHTS,
    GENRE_COLS,
    MAX_K,
)
from recommender.content_model import ContentModel
from recommender.data import validate_data
from recommender.preprocessing import Index, build_matrix
from recommender.ranking import (
    PopularityModel,
    combine,
    minmax_rows,
    normalize_components,
    top_k_indices,
)


class UnknownMovieError(KeyError):
    """Raised when a movie_id is not in the catalogue."""


def load_best_config(path: str | Path = BEST_CONFIG_PATH) -> dict:
    """Read the config written by evaluate.py, or return defaults."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "knn_signal": DEFAULT_KNN_SIGNAL,
        "svd_signal": DEFAULT_SVD_SIGNAL,
        "weights": dict(DEFAULT_WEIGHTS),
    }


def normalise_genres(genres: Iterable[str] | None) -> list[str]:
    """Match user-supplied genre names to GENRE_COLS (case-insensitive)."""
    lookup = {g.lower(): g for g in GENRE_COLS}
    matched = []
    for genre in genres or []:
        key = str(genre).strip().lower().replace(" ", "_").replace("-", "_")
        if key not in lookup:
            raise ValueError(f"Unknown genre {genre!r}. Valid genres: {GENRE_COLS}")
        matched.append(lookup[key])
    return matched


class Recommender:
    """Hybrid recommender: item-kNN + SVD + content + popularity."""

    def __init__(
        self,
        knn_signal: str = DEFAULT_KNN_SIGNAL,
        svd_signal: str = DEFAULT_SVD_SIGNAL,
        weights: dict[str, float] | None = None,
    ):
        self.knn_signal = knn_signal
        self.svd_signal = svd_signal
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        unknown = set(self.weights) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")

    # ------------------------------------------------------------------ fit
    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "Recommender":
        """Train on ALL ratings (no hold-out). Evaluation lives in evaluate.py."""
        validate_data(movies, ratings)
        self.index = Index.build(ratings["user_id"], movies["movie_id"])
        self.titles = movies.set_index("movie_id").loc[self.index.movie_ids, "title"].to_numpy()

        self.knn = ItemKNN(self.knn_signal).fit(ratings, self.index)
        self.svd = SVDModel(self.svd_signal).fit(ratings, self.index)
        self.content = ContentModel().fit(movies, self.index)
        self.popularity = PopularityModel().fit(ratings, self.index)

        self.models = {
            "knn": self.knn,
            "svd": self.svd,
            "content": self.content,
            "popularity": self.popularity,
        }
        signals = {m.signal for m in self.models.values()} | {"binary"}
        self.history = {s: build_matrix(ratings, self.index, s) for s in signals}
        self.seen = self.history["binary"] > 0
        return self

    # -------------------------------------------------------------- lookups
    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self.index.user_to_idx

    def has_movie(self, movie_id: int) -> bool:
        return int(movie_id) in self.index.movie_to_idx

    def genres_of(self, item_idx: int) -> list[str]:
        flags = self.content.genre_matrix[item_idx]
        return [g for g, on in zip(GENRE_COLS, flags, strict=True) if on]

    def _frame(self, items: np.ndarray, scores: np.ndarray, **extra) -> pd.DataFrame:
        data = {
            "movie_id": self.index.movie_ids[items],
            "title": self.titles[items],
            "genres": [self.genres_of(int(i)) for i in items],
            "score": scores,
        }
        data.update(extra)
        return pd.DataFrame(data)

    # ---------------------------------------------------------- recommenders
    def popular(self, k: int = 10, preferred_genres: Iterable[str] | None = None) -> pd.DataFrame:
        """Cold-start ranking: popularity, boosted by preferred genres."""
        k = self._clip_k(k)
        prefs = normalise_genres(preferred_genres)
        base = self.popularity.scores[None, :]
        score = minmax_rows(base, np.ones_like(base, dtype=bool))[0]
        if prefs:
            columns = [GENRE_COLS.index(g) for g in prefs]
            match = self.content.genre_matrix[:, columns].any(axis=1)
            score = score + COLD_START_GENRE_BOOST * match
        top = top_k_indices(score[None, :], k)[0]
        return self._frame(top, score[top])

    def recommend_for_user(
        self,
        user_id: int,
        k: int = 10,
        preferred_genres: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Hybrid recommendations for a known user; cold start otherwise.

        preferred_genres only affects the cold-start path.
        """
        k = self._clip_k(k)
        if not self.has_user(user_id):
            return self.popular(k, preferred_genres)
        u = self.index.user_to_idx[int(user_id)]
        seen = self.seen[u].toarray()
        raw = {
            name: model.score(self.history[model.signal][u]) for name, model in self.models.items()
        }
        normalized = normalize_components(raw, seen)
        final = combine(normalized, self.weights, seen)
        top = top_k_indices(final, k)[0]
        top = top[np.isfinite(final[0, top])]  # drop -inf if the user saw ~everything
        parts = {name: normalized[name][0, top] for name in normalized}
        return self._frame(
            top,
            final[0, top],
            components=[{name: float(parts[name][i]) for name in parts} for i in range(len(top))],
        )

    def similar_movies(self, movie_id: int, k: int = 10, mode: str = "content") -> pd.DataFrame:
        """Movies most similar to movie_id by 'content' (genres) or 'collaborative'."""
        if not self.has_movie(movie_id):
            raise UnknownMovieError(int(movie_id))
        if mode not in ("content", "collaborative"):
            raise ValueError("mode must be 'content' or 'collaborative'")
        k = self._clip_k(k)
        item = self.index.movie_to_idx[int(movie_id)]
        model = self.content if mode == "content" else self.knn
        items, scores = model.similar_items(item, k)
        return self._frame(items, scores)

    def recommend(
        self,
        user_id: int | None = None,
        movie_id: int | None = None,
        k: int = 10,
        preferred_genres: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Unified entry point: item-to-item, personalised, or cold start."""
        if movie_id is not None:
            return self.similar_movies(movie_id, k)
        if user_id is not None:
            return self.recommend_for_user(user_id, k, preferred_genres)
        return self.popular(k, preferred_genres)

    @staticmethod
    def _clip_k(k: int) -> int:
        return max(1, min(int(k), MAX_K))

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Recommender":
        """Load a saved model. Only load files you created yourself (pickle)."""
        return joblib.load(path)
