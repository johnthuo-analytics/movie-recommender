"""Train the final model on ALL ratings and save it as an artifact.

Usage:
    python train.py
Run `python evaluate.py` first if you want tuned weights (best_config.json).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recommender.config import ARTIFACT_PATH, BEST_CONFIG_PATH, DATA_DIR
from recommender.data import load_movielens
from recommender.inference import Recommender, load_best_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--config", type=Path, default=BEST_CONFIG_PATH)
    args = parser.parse_args()

    movies, ratings = load_movielens(args.data_dir)
    config = load_best_config(args.config)
    print(f"Config: {config}")

    model = Recommender(
        knn_signal=config["knn_signal"],
        svd_signal=config["svd_signal"],
        weights=config["weights"],
    ).fit(movies, ratings)
    model.save(args.out)

    print(f"Saved {args.out.resolve()}")
    print(
        f"Movies: {len(movies):,} | Ratings: {len(ratings):,} | "
        f"Users: {ratings['user_id'].nunique():,}"
    )
    sample_user = int(model.index.user_ids[0])
    print(f"\nSample recommendations for user {sample_user}:")
    print(model.recommend_for_user(sample_user, k=10)[["movie_id", "title", "score"]])


if __name__ == "__main__":
    main()
