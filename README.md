# Tech-Africa Movie Recommendation System

A hybrid movie recommender built on **MovieLens 100K**: popularity baseline,
content-based similarity, item-item collaborative filtering, SVD, a tuned hybrid
ranker, cold-start handling, offline evaluation and a FastAPI inference service.

## 1. Problem
Given a user's rating history, suggest movies they have not seen yet and are
likely to watch next. New users with no history still get sensible suggestions.

## 2. Dataset
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): 100,000 ratings
(1-5) from 943 users on 1,682 movies, plus 19 genre flags per movie.
`python train.py` and `python evaluate.py` download it automatically into `data/`.

## 3. Architecture
```text
data.py -> preprocessing.py -> content_model.py ┐
                               collaborative_model.py ├-> ranking.py (hybrid blend)
                               ranking.py (popularity) ┘         |
                                                          inference.py (Recommender)
                                                                 |
train.py -> artifacts/recommender.joblib -> api.py (FastAPI) -> Docker
evaluate.py -> reports/*.csv, model_comparison.png, best_config.json
```
Training and serving are separate: the API only loads a saved artifact.

## 4. Algorithms
| Component | Idea |
|---|---|
| Popularity | IMDB-style weighted rating; shrinks small-sample averages toward the global mean |
| Content | Cosine similarity of binary genre vectors; user score = mean similarity to liked movies |
| Item-kNN | Cosine between item columns, shrunk by co-rater count, top-50 neighbours. Rating signal (`raw`, `binary`, `centered`) chosen on validation data |
| SVD | Truncated SVD (PureSVD) with fold-in for any user history |
| Hybrid | Per-user min-max normalised scores blended with weights tuned on validation data |
| Cold start | Popularity, boosted by optional preferred genres |

## 5. Evaluation
Chronological **leave-last-out**: each user's latest rating is the test item, the
second-latest is validation. Every unseen movie is ranked and we record where the
held-out movie lands. Metrics: Precision@K, Recall@K, MAP@K, NDCG@K, MRR, plus
coverage, popularity bias and diversity. Model variants and hybrid weights are
chosen on validation; the test item is scored once. Because there is one relevant
item per user, Recall@K equals HitRate@K and Precision@K equals Recall@K / K.

Run `python evaluate.py`, then paste `reports/test_results.csv` here:

> RESULTS_TABLE: replace this line with your own results.

## 6. API
| Endpoint | Purpose |
|---|---|
| `GET /health` | service and model status |
| `GET /recommendations/user/{user_id}?k=10&preferred_genres=Drama` | personalised (hybrid), or cold start for unknown users |
| `GET /recommendations/movie/{movie_id}?k=10&mode=content\|collaborative` | similar movies; 404 if unknown |
| `GET /recommendations/popular?k=10&preferred_genres=Comedy` | popularity ranking |

Interactive docs: `http://localhost:8000/docs`.

## 7. Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # run the tests
python evaluate.py               # compare models, tune weights, write reports/
python train.py                  # train on all data, save artifacts/
uvicorn api:app --reload         # serve on http://127.0.0.1:8000
```
Docker (after `python train.py`):
```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

## 8. Limitations
- MovieLens 100K is a classroom benchmark, not live traffic; offline metrics do not guarantee online performance.
- One held-out item per user gives noisy metrics; differences between close models may not be meaningful.
- Genre-only content features are coarse: many movies share identical genre vectors.
- Dense similarity matrices suit ~1.7k movies; larger catalogues need approximate nearest neighbours.
- New movies with no ratings only get content-based scores.
- The artifact is a pickle: load only files you trained, with the same library versions.

## 9. Future improvements
Authentication and rate limiting, model versioning, monitoring and drift checks, diversity re-ranking, richer content features (TMDB metadata), implicit-feedback models (ALS/BPR), experiment tracking and scheduled retraining.
