# 🎬 Movie Recommendation System

**A production-oriented hybrid movie recommendation system combining item-kNN, SVD, content-based similarity, and popularity signals — served through FastAPI and Docker.**

> **Result:** The tuned hybrid achieves **8.38% Recall@10**, compared with **2.97% for the popularity baseline**, while keeping the recommendation pipeline explainable and suitable for API serving.

[![CI](https://github.com/johnthuo-analytics/movie-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/johnthuo-analytics/movie-recommender/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 🌐 Live Demo

**https://movie-recommender-03sk.onrender.com** · [API docs](https://movie-recommender-03sk.onrender.com/docs)

> Hosted on a free plan: after 15 minutes idle it sleeps, and the first load can take about 50 seconds.

![Movie Recommendation API docs](docs/api-docs.png)


---

## 🚀 Quick Start

```bash
git clone https://github.com/johnthuo-analytics/movie-recommender.git
cd movie-recommender

python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## ✨ Highlights

* Hybrid recommendation engine
* Collaborative filtering with SVD / PureSVD
* Item-item kNN with cosine similarity
* Genre-based content similarity
* Popularity baseline
* Cold-start recommendation strategy
* Chronological leave-last-out evaluation
* Tunable hybrid ranking
* FastAPI inference service
* Docker-ready deployment
* Automated tests with Pytest
* Ruff code-quality checks
* GitHub Actions CI
* Saved model artifacts for inference
* Training and serving separated for production-style deployment

---

# 1. Problem

Movie recommendation is fundamentally a **ranking problem**.

Given a user's historical interactions, the system must rank movies that the user is most likely to appreciate.

The project addresses two practical scenarios:

### Personalised recommendations

For users with rating history, the system combines collaborative and content-based signals to produce ranked recommendations.

### Cold-start recommendations

For users without sufficient interaction history, the system falls back to popularity-based recommendations, optionally filtered by preferred genres.

---

# 2. Dataset

The project uses the **MovieLens 100K dataset** from GroupLens.

Dataset:

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)

The dataset contains:

* **100,000 ratings**
* **943 users**
* **1,682 movies**
* Ratings from **1–5**
* **19 movie genres**

The dataset is small enough for experimentation while still providing a realistic recommendation-system workflow.

---

# 3. System Architecture

The project separates data preparation, model training, evaluation, and inference.

```text
                         ┌─────────────────────┐
                         │   MovieLens 100K    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      data.py        │
                         │  Load / validate    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   preprocessing.py  │
                         │   Transform data    │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             Content Model      kNN Model        SVD Model
                    │               │                │
                    └───────────────┼────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     ranking.py      │
                         │   Hybrid ranking    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      train.py       │
                         │    Save artifact    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ recommender.joblib  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       api.py        │
                         │      FastAPI        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      REST API       │
                         └─────────────────────┘

                         Evaluation pipeline
                                    │
                                    ▼
                              evaluate.py
                                    │
                                    ▼
                         reports/*.csv
                         model_comparison.png
                         best_config.json
```

---

# 4. Recommendation Approach

The system combines multiple recommendation signals.

## 4.1 Popularity Baseline

A popularity model provides a simple baseline and a useful fallback for users with little or no history.

The ranking incorporates rating information rather than simply sorting by raw rating count.

This gives the project a benchmark against which more sophisticated models can be compared.

---

## 4.2 Content-Based Similarity

Movies are represented using their genre information.

A movie's genre vector is compared with other movie vectors using similarity measures.

This allows the system to answer questions such as:

> "Users who like this type of movie may also like these movies."

Content similarity is particularly useful when collaborative information is limited.

---

## 4.3 Item-Item kNN

The item-based collaborative filtering model identifies movies with similar user-rating patterns.

The implementation uses:

* Cosine similarity
* Item-user interaction matrices
* Neighbour selection
* Similarity shrinkage

The model can therefore recommend movies based on relationships learned from historical user behaviour.

---

## 4.4 SVD / PureSVD

The collaborative filtering component uses a matrix-factorisation approach.

User-item interactions are represented in a latent-factor space.

The model learns:

```text
User preferences
       ↓
Latent representation
       ↓
Movie representations
       ↓
Predicted preference
       ↓
Recommendation ranking
```

The serving pipeline also supports **fold-in**, allowing an existing user's interaction history to be represented without retraining the entire model.

---

## 4.5 Hybrid Ranking

The final recommendation score combines the available signals.

The tuned configuration is:

| Component  | Weight |
| ---------- | -----: |
| SVD        |   0.90 |
| Content    |   0.10 |
| Item-kNN   |   0.00 |
| Popularity |   0.00 |

The weights are applied after score normalization.

This configuration reflects the evaluation results rather than forcing every model to contribute equally.

---

# 5. Cold-Start Strategy

A recommendation system needs a fallback for users with little or no history.

The project handles this using popularity-based recommendations.

Users can optionally provide preferred genres.

Example:

```text
User → No rating history
        ↓
Preferred genre = Drama
        ↓
Popular Drama movies
        ↓
Top-K recommendations
```

This prevents the API from failing when personalised collaborative recommendations cannot be generated.

---

# 6. Evaluation Methodology

The evaluation uses a **chronological leave-last-out strategy**.

Instead of randomly splitting user interactions, the system respects the order in which ratings occurred.

For each eligible user:

```text
Earlier interactions
        ↓
Training history

Second-latest interaction
        ↓
Validation

Latest interaction
        ↓
Test
```

This more closely represents the real-world recommendation scenario:

> Use the user's past behaviour to predict what they interact with next.

---

# 7. Evaluation Metrics

The project evaluates recommendation quality using several ranking metrics.

### Recall@K

Measures whether the relevant item appears somewhere in the top-K recommendations.

### Precision@K

Measures the proportion of recommended items that are relevant.

### MAP@K

Measures ranking quality while considering the position of relevant recommendations.

### NDCG@K

Rewards relevant items appearing higher in the recommendation list.

### MRR

Measures how early the first relevant recommendation appears.

### Coverage@K

Measures how much of the available movie catalogue is recommended.

Using multiple metrics provides a more complete view than relying on a single score.

---

# 8. Model Results

| Model      |  Recall@10 | Precision@10 |     MAP@10 |    NDCG@10 | Coverage@10 |
| ---------- | ---------: | -----------: | ---------: | ---------: | ----------: |
| Random     |     0.0053 |       0.0015 |     0.0024 |     0.0046 |      0.9958 |
| Popularity |     0.0297 |       0.0082 |     0.0130 |     0.0134 |      0.0059 |
| Content    |     0.0201 |       0.0064 |     0.0097 |     0.0104 |      0.4566 |
| Item-kNN   |     0.0827 |       0.0239 |     0.0375 |     0.0364 |      0.1950 |
| SVD        |     0.0827 |       0.0240 |     0.0375 |     0.0372 |      0.1350 |
| **Hybrid** | **0.0838** |   **0.0245** | **0.0382** | **0.0378** |  **0.2812** |

---

## 8.1 Model Comparison

![Model comparison](reports/model_comparison.png)

---

# 9. Results Interpretation

Several observations stand out.

### Hybrid vs Popularity

The tuned hybrid achieves:

```text
Hybrid Recall@10:      0.0838
Popularity Recall@10: 0.0297
```

This represents a substantial improvement over the simple popularity baseline.

### kNN and SVD

Both collaborative approaches achieve:

```text
Item-kNN: 0.0827
SVD:      0.0827
```

This shows that collaborative signals provide most of the predictive strength in this dataset.

### Hybrid improvement

The hybrid reaches:

```text
0.0838 Recall@10
```

compared with:

```text
0.0827 Recall@10
```

for SVD.

The improvement is therefore relatively small.

This is intentionally reported rather than overstated. Because the evaluation uses one held-out item per eligible user, small differences can potentially reflect evaluation noise.

The tuned weights also explain why SVD dominates the final hybrid:

```text
SVD        = 90%
Content    = 10%
kNN        = 0%
Popularity = 0%
```

---

# 10. Recommendation Coverage

Coverage provides an important counterpoint to accuracy metrics.

The models produce the following Coverage@10:

```text
Popularity   0.0059
Content      0.4566
Item-kNN     0.1950
SVD          0.1350
Hybrid       0.2812
```

The popularity model has very low catalogue coverage because it repeatedly recommends a small group of popular movies.

The content model provides substantially higher coverage.

The hybrid improves coverage relative to SVD while maintaining the strongest Recall@10 among the evaluated models.

This demonstrates why recommendation systems should not be evaluated using accuracy alone.

---

# 11. FastAPI

The trained model is served through FastAPI.

The API loads a previously trained artifact and **does not train the model during application startup**.

## API Endpoints

| Method | Endpoint                            | Purpose                                            |
| ------ | ----------------------------------- | -------------------------------------------------- |
| GET    | `/`                                 | Interactive recommendation demo                    |
| GET    | `/genres`                           | Returns valid movie genres                         |
| GET    | `/health`                           | Returns service, version, and model status         |
| GET    | `/recommendations/user/{user_id}`   | Returns personalised or cold-start recommendations |
| GET    | `/recommendations/movie/{movie_id}` | Returns similar movies                             |
| GET    | `/recommendations/popular`          | Returns popularity-based recommendations           |

---

## Query Parameters

### User recommendations

```text
/recommendations/user/{user_id}?k=10&preferred_genres=Drama
```

Parameters:

* `user_id` — MovieLens user ID
* `k` — number of recommendations
* `preferred_genres` — optional genre preference

---

### Similar movies

```text
/recommendations/movie/{movie_id}?k=10&mode=content
```

Supported modes:

```text
content
collaborative
```

Examples:

```text
/recommendations/movie/50?k=10&mode=content
```

```text
/recommendations/movie/50?k=10&mode=collaborative
```

---

### Popular recommendations

```text
/recommendations/popular?k=10&preferred_genres=Comedy
```

---

# 12. Explore the API

Start the server:

```bash
uvicorn api:app --reload
```

Then explore:

### Interactive demo

```text
http://127.0.0.1:8000/
```

### Swagger documentation

```text
http://127.0.0.1:8000/docs
```

### Health check

```text
http://127.0.0.1:8000/health
```

The Swagger interface provides an interactive way to test all API endpoints.

---

# 13. Local Development

Create the environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

The current test suite contains:

```text
43 passing
```

Run evaluation:

```bash
python evaluate.py
```

Train the recommendation artifact:

```bash
python train.py
```

Start the API:

```bash
uvicorn api:app --reload
```

---

# 14. Docker

Build the image:

```bash
docker build -t movie-recommender .
```

Run the container:

```bash
docker run -p 8000:8000 movie-recommender
```

The API is then available at:

```text
http://127.0.0.1:8000/
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 15. Testing and Code Quality

The project includes automated testing using **Pytest**.

The test suite covers important components including:

* Data processing
* Recommendation logic
* Ranking
* API behaviour
* Model interfaces
* Cold-start behaviour

Current status:

```text
43 tests passing
```

Code quality is supported by **Ruff**.

GitHub Actions runs automated checks through CI.

The goal is to keep model experimentation separate from reliable serving code.

---

# 16. Project Structure

```text
movie-recommender/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docs/
│   └── demo.png
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── recommender/
│   ├── data.py
│   ├── preprocessing.py
│   ├── content_model.py
│   ├── collaborative_model.py
│   ├── ranking.py
│   └── ...
│
├── reports/
│   ├── model_comparison.png
│   ├── *.csv
│   └── best_config.json
│
├── tests/
│   └── ...
│
├── api.py
├── train.py
├── evaluate.py
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── ruff.toml
└── README.md
```

---

# 17. Limitations

The project has several limitations.

### Dataset size

MovieLens 100K is useful for demonstrating recommendation-system concepts but is relatively small compared with production datasets.

### Sparse interactions

Most users interact with only a small portion of the catalogue.

### Limited metadata

The content model primarily relies on movie genres.

Additional information such as:

* Actors
* Directors
* Plot descriptions
* Keywords
* Release year
* User context

could improve content-based recommendations.

### Evaluation design

The leave-last-out strategy uses one held-out item per user, meaning that small differences between models should not automatically be interpreted as statistically significant improvements.

### Hybrid contribution

Although the hybrid performs best on the selected metrics, the improvement over SVD is modest.

This suggests that the next stage should focus on stronger feature engineering, evaluation robustness, and richer recommendation signals.

---

# 18. What I Would Improve Next

Several improvements would make the system stronger.

### 1. Add richer movie metadata

Integrate:

* Movie descriptions
* Actors
* Directors
* Keywords
* Release year

This would allow a more expressive content model.

### 2. Improve hybrid optimisation

Instead of manually tuning a small set of weights, experiment with:

* Learning-to-rank
* Logistic regression
* Gradient boosting
* Pairwise ranking models

### 3. Improve evaluation robustness

Run:

* Multiple chronological splits
* Multiple random seeds
* Bootstrap confidence intervals
* Per-user evaluation
* Statistical significance testing

### 4. Add explainability

Return recommendation reasons such as:

```text
Recommended because you liked:

Movie A
Movie B
Movie C
```

or:

```text
Recommended because it matches your preferred genres:

Drama, Romance
```

### 5. Improve production deployment

Potential next steps include:

* Model versioning
* Structured logging
* Monitoring
* API authentication
* Request metrics
* Cloud deployment
* CI/CD deployment pipeline

---

# 19. Key Takeaways

This project demonstrates an end-to-end recommendation workflow:

```text
Raw data
   ↓
Data preparation
   ↓
Multiple recommendation models
   ↓
Evaluation
   ↓
Hyperparameter / weight tuning
   ↓
Saved model artifact
   ↓
FastAPI inference
   ↓
Docker deployment
```

The main lessons are:

* Collaborative filtering provides the strongest predictive signal on this dataset.
* SVD and item-kNN perform similarly.
* The tuned hybrid achieves the best overall Recall@10.
* Content-based recommendations improve catalogue coverage.
* Popularity is useful as a simple baseline and cold-start fallback.
* Evaluation methodology matters when interpreting small model differences.
* Separating training from inference makes the application more production-oriented.

---

# 20. Why This Project Matters

The project goes beyond simply training a recommendation model.

It demonstrates the complete path from:

**data → modelling → evaluation → API → deployment**

It combines machine learning with software engineering practices including:

* Modular architecture
* Reproducible training
* Model persistence
* Automated testing
* Code quality checks
* REST API development
* Dockerisation
* Continuous integration

This makes the project representative of a practical machine-learning workflow rather than a standalone notebook experiment.

---

## 📄 License

This project is licensed under the **MIT License**.
