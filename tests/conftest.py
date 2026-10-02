import pytest

from tests.synthetic import make_synthetic


@pytest.fixture(scope="session")
def data():
    """Small synthetic (movies, ratings) pair in MovieLens format."""
    return make_synthetic()


@pytest.fixture(scope="session")
def model(data):
    """A fully trained Recommender (imported lazily: it is built in Step 7)."""
    from recommender.inference import Recommender

    movies, ratings = data
    return Recommender().fit(movies, ratings)
