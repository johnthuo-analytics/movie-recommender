import numpy as np

from recommender.content_model import ContentModel
from recommender.preprocessing import Index


def test_content_similarity_properties(data):
    movies, ratings = data
    index = Index.build(ratings.user_id, movies.movie_id)
    model = ContentModel().fit(movies, index)
    assert model.similarity.shape == (index.n_items, index.n_items)
    assert np.allclose(model.similarity, model.similarity.T)
    items, scores = model.similar_items(0, 5)
    assert 0 not in items and len(items) == 5
    assert np.all(np.diff(scores) <= 1e-12)  # sorted best-first
