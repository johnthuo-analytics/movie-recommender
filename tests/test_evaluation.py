import numpy as np

from recommender.evaluation import (
    catalog_coverage,
    intra_list_diversity,
    mean_popularity,
    rank_of_target,
    ranking_metrics,
)


def test_rank_of_target_basic():
    scores = np.array([[0.9, 0.5, 0.1, 0.7]])
    seen = np.zeros((1, 4), dtype=bool)
    assert rank_of_target(scores, seen, np.array([3]))[0] == 2  # 0.9 beats 0.7


def test_rank_of_target_ignores_seen_items():
    scores = np.array([[0.9, 0.5, 0.1, 0.7]])
    seen = np.array([[True, False, False, False]])
    assert rank_of_target(scores, seen, np.array([3]))[0] == 1


def test_rank_ties_share_positions():
    scores = np.ones((1, 5))
    seen = np.zeros((1, 5), dtype=bool)
    assert rank_of_target(scores, seen, np.array([0]))[0] == 3.0  # 1 + 4/2


def test_metrics_perfect_and_miss():
    perfect = ranking_metrics(np.array([1.0]), ks=(10,))
    assert perfect["Recall@10"] == 1 and perfect["NDCG@10"] == 1
    assert perfect["Precision@10"] == 0.1 and perfect["MAP@10"] == 1
    miss = ranking_metrics(np.array([50.0]), ks=(10,))
    assert miss["Recall@10"] == 0 and miss["NDCG@10"] == 0


def test_ndcg_formula():
    out = ranking_metrics(np.array([3.0]), ks=(10,))
    assert np.isclose(out["NDCG@10"], 1 / np.log2(4))
    assert np.isclose(out["MAP@10"], 1 / 3)


def test_coverage_popularity_diversity():
    top = np.array([[0, 1], [1, 2]])
    assert catalog_coverage(top, 10) == 0.3
    assert mean_popularity(top, np.array([10, 20, 30])) == 20
    sim = np.eye(3)
    assert intra_list_diversity(top, sim) == 1.0
