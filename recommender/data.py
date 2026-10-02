"""Loading and validating the MovieLens 100K dataset."""

from __future__ import annotations

import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

from recommender.config import (
    DATA_DIR,
    GENRE_COLS,
    MOVIE_COLS,
    MOVIELENS_URL,
    RATING_COLS,
)


def download_movielens(data_dir: str | Path = DATA_DIR) -> Path:
    """Download and extract MovieLens 100K. Returns the extracted folder."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ml-100k.zip"

    if not zip_path.exists():
        context = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(MOVIELENS_URL, context=context) as response:
            with open(zip_path, "wb") as output:
                output.write(response.read())

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)

    return data_dir / "ml-100k"


def load_movielens(
    data_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (movies, ratings). Downloads the data if it is not on disk."""
    data_dir = Path(data_dir)
    folder = data_dir / "ml-100k"

    if not (folder / "u.data").exists():
        folder = download_movielens(data_dir)

    ratings = pd.read_csv(
        folder / "u.data",
        sep="\t",
        names=RATING_COLS,
    )

    movies = pd.read_csv(
        folder / "u.item",
        sep="|",
        names=MOVIE_COLS,
        encoding="latin-1",
    )

    return movies, ratings


def validate_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Fail loudly if the data breaks an assumption the models rely on."""
    missing_r = set(RATING_COLS) - set(ratings.columns)

    if missing_r:
        raise ValueError(f"ratings is missing columns: {sorted(missing_r)}")

    missing_m = {"movie_id", "title", *GENRE_COLS} - set(movies.columns)

    if missing_m:
        raise ValueError(f"movies is missing columns: {sorted(missing_m)}")

    if ratings[["user_id", "movie_id", "rating"]].isna().any().any():
        raise ValueError("ratings contains missing values")

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("ratings must be between 1 and 5")

    if ratings.duplicated(subset=["user_id", "movie_id"]).any():
        raise ValueError("duplicate (user_id, movie_id) pairs found")

    if movies["movie_id"].duplicated().any():
        raise ValueError("duplicate movie_id values in movies table")

    if not ratings["movie_id"].isin(movies["movie_id"]).all():
        raise ValueError("ratings reference movie_ids missing from movies")