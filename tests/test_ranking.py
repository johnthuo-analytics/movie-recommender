import numpy as np
import pandas as pd

from recommender.preprocessing import Index
from recommender.ranking import (
    PopularityModel,
    combine,
    minmax_rows,
    normalize_components,
    top_k_indices,
)


def test_popularity_prefers_well_supported_movies():
    # movie 1: twenty 5-star ratings | movie 2: a single 5-star rating
    # movie 3: twenty 2-star ratings (pulls the global mean down)
    ratings = pd.DataFrame(
        {
            "user_id": list(range(1, 42)),
            "movie_id": [1] * 20 + [2] + [3] * 20,
            "rating": [5] * 20 + [5] + [2] * 20,
            "timestamp": range(41),
        }
    )
    index = Index.build(ratings.user_id, [1, 2, 3])
    model = PopularityModel().fit(ratings, index)
    # Same average (5.0), but shrinkage trusts the movie with more evidence
    assert model.scores[0] > model.scores[1] > model.scores[2]


def test_minmax_rows_ignores_seen_items():
    scores = np.array([[1.0, 2.0, 3.0, 100.0]])
    candidates = np.array([[True, True, True, False]])
    out = minmax_rows(scores, candidates)
    assert out[0].tolist() == [0.0, 0.5, 1.0, 0.0]


def test_minmax_rows_constant_row_is_zero():
    out = minmax_rows(np.ones((1, 4)), np.ones((1, 4), dtype=bool))
    assert (out == 0).all()


def test_combine_masks_seen_and_applies_weights():
    seen = np.array([[False, False, True]])
    comps = {"a": np.array([[0.0, 1.0, 5.0]]), "b": np.array([[1.0, 0.0, 5.0]])}
    norm = normalize_components(comps, seen)
    final = combine(norm, {"a": 0.75, "b": 0.25}, seen)
    assert final[0, 2] == -np.inf
    assert final[0, 1] > final[0, 0]


def test_top_k_indices_sorted():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k_indices(scores, 3)[0].tolist() == [1, 3, 2]
