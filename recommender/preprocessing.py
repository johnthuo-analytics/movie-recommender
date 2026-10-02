"""Splitting, ID indexing and interaction-matrix construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recommender.config import LIKE_THRESHOLD

SIGNALS = ("raw", "binary", "centered", "liked")


@dataclass(frozen=True)
class Index:
    """Maps external user/movie IDs to matrix row/column positions."""

    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_to_idx: dict
    movie_to_idx: dict

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.movie_ids)

    @classmethod
    def build(cls, user_ids, movie_ids) -> "Index":
        users = np.sort(np.unique(np.asarray(user_ids)))
        movies = np.sort(np.unique(np.asarray(movie_ids)))
        return cls(
            user_ids=users,
            movie_ids=movies,
            user_to_idx={int(u): i for i, u in enumerate(users)},
            movie_to_idx={int(m): i for i, m in enumerate(movies)},
        )


@dataclass(frozen=True)
class Splits:
    """Chronological leave-last-out splits.

    train_full = train + validation item; used for the final test evaluation.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_full: pd.DataFrame


def filter_active_users(ratings: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    counts = ratings.groupby("user_id")["rating"].transform("size")
    return ratings[counts >= min_ratings].copy()


def leave_last_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent rating. Returns (rest, held_out)."""
    ordered = ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
    last_rows = ordered.groupby("user_id").tail(1).index
    held_out = ordered.loc[last_rows].reset_index(drop=True)
    rest = ordered.drop(last_rows).reset_index(drop=True)
    return rest, held_out


def temporal_split(ratings: pd.DataFrame, min_ratings: int) -> Splits:
    """Last item -> test, second-to-last -> validation, the rest -> train."""
    eligible = filter_active_users(ratings, min_ratings)
    train_full, test = leave_last_out(eligible)
    train, val = leave_last_out(train_full)
    return Splits(train=train, val=val, test=test, train_full=train_full)


def build_matrix(ratings: pd.DataFrame, index: Index, signal: str = "raw") -> csr_matrix:
    """Build a (users x items) sparse matrix from a ratings table.

    signal:
      raw      the 1-5 rating
      binary   1 if the user rated the movie (implicit feedback)
      centered rating minus that user's mean rating
      liked    1 if rating >= LIKE_THRESHOLD
    Rows for users/movies outside the index are ignored.
    """
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")
    rows = ratings["user_id"].map(index.user_to_idx)
    cols = ratings["movie_id"].map(index.movie_to_idx)
    keep = rows.notna() & cols.notna()
    frame = ratings[keep]
    rows = rows[keep].astype(int).to_numpy()
    cols = cols[keep].astype(int).to_numpy()

    if signal == "raw":
        values = frame["rating"].to_numpy(dtype=float)
    elif signal == "binary":
        values = np.ones(len(frame))
    elif signal == "liked":
        values = (frame["rating"] >= LIKE_THRESHOLD).to_numpy(dtype=float)
    else:  # centered
        user_mean = frame.groupby("user_id")["rating"].transform("mean")
        values = (frame["rating"] - user_mean).to_numpy(dtype=float)

    return csr_matrix((values, (rows, cols)), shape=(index.n_users, index.n_items))
