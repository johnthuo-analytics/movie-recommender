import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture()
def client(model):
    with TestClient(create_app(model=model)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["version"]


def test_user_recommendations(client):
    r = client.get("/recommendations/user/1?k=5")
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "hybrid"
    assert len(body["items"]) == 5
    assert {"movie_id", "title", "genres", "score"} <= set(body["items"][0])


def test_unknown_user_is_cold_start(client):
    body = client.get("/recommendations/user/99999?k=3").json()
    assert body["strategy"] == "cold_start"
    assert len(body["items"]) == 3


def test_movie_recommendations_and_404(client):
    ok = client.get("/recommendations/movie/1?k=4&mode=collaborative")
    assert ok.status_code == 200
    assert ok.json()["strategy"] == "similar_collaborative"
    assert client.get("/recommendations/movie/99999").status_code == 404


def test_popular_with_genres_and_bad_genre(client):
    ok = client.get("/recommendations/popular?k=5&preferred_genres=Drama&preferred_genres=Comedy")
    assert ok.status_code == 200
    assert client.get("/recommendations/popular?preferred_genres=bogus").status_code == 400


@pytest.mark.parametrize("k", [0, 51])
def test_k_validation(client, k):
    assert client.get(f"/recommendations/user/1?k={k}").status_code == 422


def test_503_when_model_missing(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/recommendations/user/1").status_code == 503


def test_demo_page_and_genres(client):
    page = client.get("/")
    assert page.status_code == 200 and "Movie Recommender" in page.text
    assert "Drama" in client.get("/genres").json()["genres"]


def test_docs_page_is_styled(client):
    page = client.get("/docs")
    assert page.status_code == 200 and "swagger-ui" in page.text and "7c5cff" in page.text
