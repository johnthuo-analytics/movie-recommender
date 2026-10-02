"""Central configuration. Every constant and default lives here."""

from pathlib import Path

# ---- Paths -----------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "recommender.joblib"
REPORTS_DIR = Path("reports")
BEST_CONFIG_PATH = REPORTS_DIR / "best_config.json"  # written by evaluate.py

# ---- MovieLens schema ------------------------------------------------------
RATING_COLS = ["user_id", "movie_id", "rating", "timestamp"]
GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western",
]  # fmt: skip
MOVIE_COLS = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
    *GENRE_COLS,
]  # fmt: skip

# ---- Experiment settings ---------------------------------------------------
RANDOM_STATE = 42
MIN_USER_RATINGS = 5  # users below this are treated as cold-start
LIKE_THRESHOLD = 4  # rating >= this counts as "liked"
EVAL_KS = (5, 10, 20)  # K values reported by the evaluation module

# ---- Model hyper-parameters ------------------------------------------------
KNN_NEIGHBORS = 50  # neighbours kept per item
KNN_SHRINKAGE = 10.0  # damps similarities built from few common raters
SVD_FACTORS = 20  # latent factors for the SVD model
POPULARITY_QUANTILE = 0.80  # "m" in the IMDB-style weighted rating

# ---- Hybrid ranking --------------------------------------------------------
COMPONENTS = ("knn", "svd", "content", "popularity")
DEFAULT_WEIGHTS = {"knn": 0.4, "svd": 0.2, "content": 0.2, "popularity": 0.2}
DEFAULT_KNN_SIGNAL = "centered"
DEFAULT_SVD_SIGNAL = "centered"
COLD_START_GENRE_BOOST = 0.5  # added to the normalised popularity score

# ---- API -------------------------------------------------------------------
DEFAULT_K = 10
MAX_K = 50
