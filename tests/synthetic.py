"""Small synthetic dataset in MovieLens 100K format (for tests and demos).

Users belong to taste groups; each group prefers the movies of one genre, so a
working recommender has real structure to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recommender.config import GENRE_COLS, MOVIE_COLS

TASTE_GENRES = ["Action", "Comedy", "Drama", "Romance"]


def make_synthetic(n_users: int = 80, n_items: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    flags = np.zeros((n_items, len(GENRE_COLS)), dtype=int)
    primary = rng.integers(0, len(TASTE_GENRES), n_items)
    for i, g in enumerate(primary):
        flags[i, GENRE_COLS.index(TASTE_GENRES[g])] = 1
        if rng.random() < 0.3:  # some movies have a second genre
            flags[i, GENRE_COLS.index("Thriller")] = 1
    movies = pd.DataFrame(flags, columns=GENRE_COLS)
    movies.insert(0, "movie_id", np.arange(1, n_items + 1))
    movies.insert(1, "title", [f"Movie {i} ({1990 + i % 30})" for i in range(1, n_items + 1)])
    movies["release_date"] = "01-Jan-1995"
    movies["video_release_date"] = np.nan
    movies["imdb_url"] = ""
    movies = movies[MOVIE_COLS]

    popularity = rng.random(n_items) ** 2 + 0.05
    rows = []
    for user in range(1, n_users + 1):
        group = user % len(TASTE_GENRES)
        n_rated = int(rng.integers(15, 35))
        chosen_p = popularity * np.where(primary == group, 4.0, 1.0)
        chosen = rng.choice(n_items, size=n_rated, replace=False, p=chosen_p / chosen_p.sum())
        times = np.sort(rng.integers(880_000_000, 890_000_000, n_rated))
        for item, ts in zip(chosen, times, strict=True):
            base = 4.3 if primary[item] == group else 2.5
            rating = int(np.clip(round(base + rng.normal(0, 0.8)), 1, 5))
            rows.append((user, item + 1, rating, int(ts)))
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return movies, ratings


def write_movielens_files(folder: Path, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Write u.data / u.item exactly the way MovieLens 100K stores them."""
    folder.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(folder / "u.data", sep="\t", header=False, index=False)
    movies.to_csv(folder / "u.item", sep="|", header=False, index=False, encoding="latin-1")
