import numpy as np

from recommender.collaborative_model import ItemKNN, SVDModel, prune_top_k
from recommender.preprocessing import Index, build_matrix


def _setup(data):
    movies, ratings = data
    return ratings, Index.build(ratings.user_id, movies.movie_id)


def test_prune_top_k_keeps_largest():
    sim = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
    pruned = prune_top_k(sim, 1)
    assert (pruned > 0).sum(axis=1).tolist() == [1, 1]
    assert pruned[0, 1] == 0.9 and pruned[1, 2] == 0.8


def test_item_knn_similarity_is_pruned_and_nonnegative(data):
    ratings, index = _setup(data)
    model = ItemKNN("centered", n_neighbors=10).fit(ratings, index)
    sim = model.similarity.toarray()
    assert (sim >= 0).all()
    assert np.allclose(np.diag(sim), 0)
    assert ((sim > 0).sum(axis=1) <= 10).all()


def test_item_knn_and_svd_score_shapes(data):
    ratings, index = _setup(data)
    for model in (ItemKNN("binary"), SVDModel("binary", n_factors=5)):
        model.fit(ratings, index)
        rows = build_matrix(ratings, index, model.signal)
        assert model.score(rows).shape == (index.n_users, index.n_items)


def test_svd_is_reproducible(data):
    ratings, index = _setup(data)
    a = SVDModel("raw", n_factors=5).fit(ratings, index)
    b = SVDModel("raw", n_factors=5).fit(ratings, index)
    rows = build_matrix(ratings, index, "raw")
    assert np.allclose(a.score(rows), b.score(rows))
