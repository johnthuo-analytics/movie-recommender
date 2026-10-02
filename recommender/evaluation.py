"""Offline evaluation for leave-last-out ranking.

Each user has exactly ONE held-out movie. We rank every movie the user has not
already seen and record where the held-out movie lands. With a single relevant
item per user:
  Recall@K  = HitRate@K (did the item make the top K?)
  Precision@K = Recall@K / K
  MAP@K     = mean of 1/rank for items inside the top K
  NDCG@K    = mean of 1/log2(rank + 1) for items inside the top K
"""

from __future__ import annotations

import numpy as np

from recommender.config import EVAL_KS


def rank_of_target(scores: np.ndarray, seen: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    """1-based rank of each user's target among unseen items.

    Ties share the average position, so a model that gives every movie the same
    score is not rewarded for luck.
    """
    masked = np.where(seen, -np.inf, scores)
    rows = np.arange(len(target_idx))
    target = masked[rows, target_idx][:, None]
    greater = (masked > target).sum(axis=1)
    equal_others = (masked == target).sum(axis=1) - 1
    return 1.0 + greater + equal_others / 2.0


def ranking_metrics(ranks: np.ndarray, ks=EVAL_KS) -> dict[str, float]:
    """Aggregate per-user ranks into Precision/Recall/MAP/NDCG at each K."""
    ranks = np.asarray(ranks, dtype=float)
    out: dict[str, float] = {}
    for k in ks:
        hit = ranks <= k
        out[f"Precision@{k}"] = float(hit.mean() / k)
        out[f"Recall@{k}"] = float(hit.mean())
        out[f"MAP@{k}"] = float(np.where(hit, 1.0 / ranks, 0.0).mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean())
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def catalog_coverage(top_idx: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K."""
    return float(len(np.unique(top_idx)) / n_items)


def mean_popularity(top_idx: np.ndarray, item_counts: np.ndarray) -> float:
    """Average number of training ratings of recommended items.

    High values mean the model mostly recommends blockbusters (popularity bias).
    """
    return float(item_counts[top_idx].mean())


def intra_list_diversity(top_idx: np.ndarray, similarity: np.ndarray) -> float:
    """Mean (1 - pairwise similarity) inside each recommendation list."""
    k = top_idx.shape[1]
    if k < 2:
        return 0.0
    off_diagonal = ~np.eye(k, dtype=bool)
    values = []
    for row in top_idx:
        block = similarity[np.ix_(row, row)]
        values.append(1.0 - block[off_diagonal].mean())
    return float(np.mean(values))
