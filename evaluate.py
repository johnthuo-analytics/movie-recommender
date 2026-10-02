"""Offline evaluation: compare models, tune hybrid weights, write reports.

Protocol (chronological leave-last-out, one held-out movie per user):
  1. Validation phase: fit on `train`, score the validation item.
     Pick the best signal variant for kNN and SVD, then tune hybrid weights.
  2. Test phase: refit on `train + validation`, score the test item ONCE,
     using the settings chosen in step 1. Nothing is tuned on the test set.
The chosen settings are saved to reports/best_config.json, which train.py reads.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.collaborative_model import ItemKNN, SVDModel
from recommender.config import (
    BEST_CONFIG_PATH,
    COMPONENTS,
    DATA_DIR,
    MIN_USER_RATINGS,
    REPORTS_DIR,
)
from recommender.content_model import ContentModel
from recommender.data import load_movielens, validate_data
from recommender.evaluation import (
    catalog_coverage,
    intra_list_diversity,
    mean_popularity,
    rank_of_target,
    ranking_metrics,
)
from recommender.preprocessing import Index, build_matrix, temporal_split
from recommender.ranking import (
    PopularityModel,
    RandomModel,
    combine,
    normalize_components,
    top_k_indices,
)

KNN_SIGNALS = ("centered", "binary", "raw")
SVD_SIGNALS = ("centered", "binary", "raw")
SELECT_METRIC = "NDCG@10"


def score_models(models: dict, history: pd.DataFrame, index: Index) -> dict:
    """Score every user against every item with each model."""
    cache: dict[str, object] = {}
    out = {}
    for name, model in models.items():
        if model.signal not in cache:
            cache[model.signal] = build_matrix(history, index, model.signal)
        out[name] = model.score(cache[model.signal])
    return out


def targets_for(held_out: pd.DataFrame, index: Index) -> np.ndarray:
    movie_ids = held_out.set_index("user_id").loc[index.user_ids, "movie_id"]
    return movie_ids.map(index.movie_to_idx).to_numpy(dtype=int)


def weight_grid(names, step: float = 0.1):
    """Every weight combination that sums to 1 in multiples of `step`."""
    n = round(1 / step)
    for combo in itertools.product(range(n + 1), repeat=len(names)):
        if sum(combo) == n:
            yield {name: c / n for name, c in zip(names, combo, strict=True)}


def fit_components(train, movies, index, knn_signal, svd_signal) -> dict:
    return {
        "knn": ItemKNN(knn_signal).fit(train, index),
        "svd": SVDModel(svd_signal).fit(train, index),
        "content": ContentModel().fit(movies, index),
        "popularity": PopularityModel().fit(train, index),
    }


def evaluate_scores(scores, seen, targets) -> dict:
    return ranking_metrics(rank_of_target(scores, seen, targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--config-out", type=Path, default=BEST_CONFIG_PATH)
    parser.add_argument("--step", type=float, default=0.1, help="weight grid step")
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)

    movies, ratings = load_movielens(args.data_dir)
    validate_data(movies, ratings)
    splits = temporal_split(ratings, MIN_USER_RATINGS)
    index = Index.build(splits.train_full["user_id"], movies["movie_id"])
    print(f"Users: {index.n_users} | Movies: {index.n_items}")

    # ------------------------------------------------ 1. validation phase
    print("\n[1/3] Validation: comparing model variants")
    seen_val = build_matrix(splits.train, index, "binary").toarray() > 0
    target_val = targets_for(splits.val, index)

    variants = {"random": RandomModel(), "popularity": PopularityModel().fit(splits.train, index)}
    variants["content"] = ContentModel().fit(movies, index)
    for s in KNN_SIGNALS:
        variants[f"knn_{s}"] = ItemKNN(s).fit(splits.train, index)
    for s in SVD_SIGNALS:
        variants[f"svd_{s}"] = SVDModel(s).fit(splits.train, index)

    val_scores = score_models(variants, splits.train, index)
    val_table = pd.DataFrame(
        {n: evaluate_scores(s, seen_val, target_val) for n, s in val_scores.items()}
    ).T
    print(val_table[["Recall@10", "MAP@10", "NDCG@10", "MRR"]].round(4))

    best_knn = max(KNN_SIGNALS, key=lambda s: val_table.loc[f"knn_{s}", SELECT_METRIC])
    best_svd = max(SVD_SIGNALS, key=lambda s: val_table.loc[f"svd_{s}", SELECT_METRIC])
    print(f"\nBest kNN signal: {best_knn} | best SVD signal: {best_svd}")

    # ---------------------------------------- 2. tune hybrid weights (val)
    print("\n[2/3] Tuning hybrid weights on the validation set")
    components = {
        "knn": val_scores[f"knn_{best_knn}"],
        "svd": val_scores[f"svd_{best_svd}"],
        "content": val_scores["content"],
        "popularity": val_scores["popularity"],
    }
    normalized = normalize_components(components, seen_val)
    best_weights, best_value = None, -1.0
    for weights in weight_grid(COMPONENTS, args.step):
        final = combine(normalized, weights, seen_val)
        value = ranking_metrics(rank_of_target(final, seen_val, target_val), ks=(10,))[
            SELECT_METRIC
        ]
        if value > best_value:
            best_weights, best_value = weights, value
    print(f"Best weights: {best_weights}  (validation {SELECT_METRIC} = {best_value:.4f})")

    config = {"knn_signal": best_knn, "svd_signal": best_svd, "weights": best_weights}
    args.config_out.write_text(json.dumps(config, indent=2))
    val_table.round(4).to_csv(args.reports_dir / "validation_results.csv")

    # ------------------------------------------------ 3. test phase (once)
    print("\n[3/3] Test: refit on train+validation, score the held-out test item")
    seen_test = build_matrix(splits.train_full, index, "binary").toarray() > 0
    target_test = targets_for(splits.test, index)
    parts = fit_components(splits.train_full, movies, index, best_knn, best_svd)
    models = {"random": RandomModel(), **parts}
    raw = score_models(models, splits.train_full, index)
    norm = normalize_components({n: raw[n] for n in COMPONENTS}, seen_test)

    finals = {
        "random": raw["random"],
        "popularity": raw["popularity"],
        "content": raw["content"],
        f"knn ({best_knn})": raw["knn"],
        f"svd ({best_svd})": raw["svd"],
        "HYBRID (tuned)": combine(norm, best_weights, seen_test),
    }
    for dropped in COMPONENTS:  # ablation: remove one component, renormalise
        rest = {n: w for n, w in best_weights.items() if n != dropped}
        total = sum(rest.values())
        if total > 0:
            rest = {n: w / total for n, w in rest.items()}
            finals[f"hybrid without {dropped}"] = combine(norm, rest, seen_test)

    rows = {}
    for name, scores in finals.items():
        metrics = evaluate_scores(scores, seen_test, target_test)
        top = top_k_indices(scores, 10)
        metrics["Coverage@10"] = catalog_coverage(top, index.n_items)
        metrics["AvgPopularity@10"] = mean_popularity(top, parts["popularity"].item_counts)
        metrics["Diversity@10"] = intra_list_diversity(top, parts["content"].similarity)
        rows[name] = metrics
    test_table = pd.DataFrame(rows).T.round(4)
    test_table.to_csv(args.reports_dir / "test_results.csv")
    show = ["Precision@10", "Recall@10", "MAP@10", "NDCG@10", "MRR", "Coverage@10", "Diversity@10"]
    print(test_table[show])

    plot_results(test_table, args.reports_dir / "model_comparison.png")
    print(f"\nSaved config -> {args.config_out}")
    print(
        f"Saved reports -> {args.reports_dir}/ "
        "(validation_results.csv, test_results.csv, model_comparison.png)"
    )


def plot_results(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_rows = [r for r in table.index if not r.startswith("hybrid without")]
    ax = table.loc[main_rows, ["Recall@10", "MAP@10", "NDCG@10"]].plot(kind="bar", figsize=(9, 4.5))
    ax.set_title("Test-set comparison (leave-last-out, K=10)")
    ax.set_ylabel("score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
