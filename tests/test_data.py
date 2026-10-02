import pandas as pd
import pytest

from recommender.data import load_movielens, validate_data
from tests.synthetic import write_movielens_files


def test_load_movielens_parses_files(tmp_path, data):
    movies, ratings = data
    write_movielens_files(tmp_path / "ml-100k", movies, ratings)
    loaded_movies, loaded_ratings = load_movielens(tmp_path)
    assert len(loaded_movies) == len(movies)
    assert len(loaded_ratings) == len(ratings)
    validate_data(loaded_movies, loaded_ratings)


def test_validate_rejects_bad_rating(data):
    movies, ratings = data
    bad = ratings.copy()
    bad.loc[0, "rating"] = 9
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_data(movies, bad)


def test_validate_rejects_duplicates_and_orphans(data):
    movies, ratings = data
    with pytest.raises(ValueError, match="duplicate"):
        validate_data(movies, pd.concat([ratings, ratings.head(1)]))
    orphan = ratings.copy()
    orphan.loc[0, "movie_id"] = 99999
    with pytest.raises(ValueError, match="missing from movies"):
        validate_data(movies, orphan)
