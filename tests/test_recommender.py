import numpy as np
import pytest

from recommender.inference import Recommender, UnknownMovieError


def test_recommend_for_user_returns_k_unseen_movies(model, data):
    _, ratings = data
    user = 1
    out = model.recommend_for_user(user, k=10)
    assert len(out) == 10
    seen = set(ratings.loc[ratings.user_id == user, "movie_id"])
    assert not seen & set(out["movie_id"])
    assert out["score"].is_monotonic_decreasing
    assert set(out.loc[0, "components"]) == {"knn", "svd", "content", "popularity"}


def test_unknown_user_gets_cold_start(model):
    out = model.recommend_for_user(user_id=99999, k=5)
    assert len(out) == 5
    assert "components" not in out.columns


def test_cold_start_genre_preference_changes_results(model):
    plain = model.popular(10)
    drama = model.popular(10, preferred_genres=["drama"])
    assert set(plain.movie_id) != set(drama.movie_id)
    assert all("Drama" in genres for genres in drama.genres.head(5))


def test_invalid_genre_raises(model):
    with pytest.raises(ValueError, match="Unknown genre"):
        model.popular(5, preferred_genres=["not-a-genre"])


@pytest.mark.parametrize("mode", ["content", "collaborative"])
def test_similar_movies_excludes_itself(model, mode):
    out = model.similar_movies(1, k=5, mode=mode)
    assert 1 not in set(out.movie_id)
    assert 0 < len(out) <= 5


def test_unknown_movie_raises(model):
    with pytest.raises(UnknownMovieError):
        model.similar_movies(99999)


def test_k_is_clipped(model):
    assert len(model.popular(k=10_000)) == 50


def test_unified_recommend_routes(model):
    assert len(model.recommend(k=3)) == 3
    assert len(model.recommend(user_id=1, k=3)) == 3
    assert len(model.recommend(movie_id=1, k=3)) == 3


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = Recommender.load(path)
    a = model.recommend_for_user(2, k=5)
    b = loaded.recommend_for_user(2, k=5)
    assert a["movie_id"].tolist() == b["movie_id"].tolist()
    assert np.allclose(a["score"], b["score"])


def test_personalisation_beats_random_on_synthetic_structure(model):
    """Users in a taste group should mostly be shown that group's genre."""
    genre_of = {"Action": 0, "Comedy": 1, "Drama": 2, "Romance": 3}
    hits = []
    for user in range(1, 41):
        out = model.recommend_for_user(user, k=10)
        target = list(genre_of)[user % 4]
        hits.append(np.mean([target in g for g in out.genres]))
    assert np.mean(hits) > 0.5  # random would be ~0.25
