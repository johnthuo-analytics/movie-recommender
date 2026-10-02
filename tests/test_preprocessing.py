import numpy as np
import pytest

from recommender.preprocessing import (
    Index,
    build_matrix,
    filter_active_users,
    leave_last_out,
    temporal_split,
)


def test_leave_last_out_holds_out_latest(data):
    _, ratings = data
    rest, held = leave_last_out(ratings)
    assert held["user_id"].is_unique
    assert len(rest) + len(held) == len(ratings)
    latest = ratings.groupby("user_id")["timestamp"].max()
    assert (held.set_index("user_id")["timestamp"] == latest).all()


def test_temporal_split_has_no_overlap(data):
    _, ratings = data
    s = temporal_split(ratings, min_ratings=5)

    def key(df):
        return set(zip(df.user_id, df.movie_id, strict=True))

    assert not key(s.train) & key(s.val)
    assert not key(s.train_full) & key(s.test)
    assert len(s.train) + len(s.val) == len(s.train_full)


def test_filter_active_users_drops_sparse_users(data):
    _, ratings = data
    assert filter_active_users(ratings, min_ratings=10**6).empty


def test_build_matrix_signals(data):
    movies, ratings = data
    index = Index.build(ratings.user_id, movies.movie_id)
    binary = build_matrix(ratings, index, "binary")
    raw = build_matrix(ratings, index, "raw")
    centered = build_matrix(ratings, index, "centered")
    assert binary.shape == (index.n_users, index.n_items)
    assert binary.nnz == len(ratings)
    assert raw.max() <= 5
    # each user's centered ratings sum to ~0
    assert np.allclose(np.asarray(centered.sum(axis=1)).ravel(), 0, atol=1e-8)
    with pytest.raises(ValueError):
        build_matrix(ratings, index, "nonsense")
