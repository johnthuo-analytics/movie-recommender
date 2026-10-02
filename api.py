"""FastAPI inference layer. Serves a pre-trained artifact; never trains."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from recommender.config import ARTIFACT_PATH, DEFAULT_K, MAX_K
from recommender.inference import Recommender, UnknownMovieError


class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float
    components: dict[str, float] | None = None


class RecommendationResponse(BaseModel):
    strategy: Literal["hybrid", "cold_start", "similar_content", "similar_collaborative", "popular"]
    items: list[Recommendation]


def _records(frame) -> list[Recommendation]:
    return [Recommendation(**row) for row in frame.to_dict(orient="records")]


def create_app(model: Recommender | None = None, model_path: str | Path | None = None) -> FastAPI:
    """Application factory. Pass `model` directly in tests."""
    path = Path(model_path or os.environ.get("MODEL_PATH", ARTIFACT_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model
        if app.state.model is None and path.exists():
            app.state.model = Recommender.load(path)
        yield

    app = FastAPI(title="Movie Recommendation API", version="2.0.0", lifespan=lifespan)

    def get_model(request: Request) -> Recommender:
        loaded = request.app.state.model
        if loaded is None:
            raise HTTPException(503, "Model artifact not loaded. Run `python train.py` first.")
        return loaded

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.model is not None}

    @app.get("/recommendations/popular", response_model=RecommendationResponse)
    def popular(
        request: Request,
        k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
        preferred_genres: list[str] | None = Query(None),
    ):
        rec = get_model(request)
        try:
            frame = rec.popular(k, preferred_genres)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RecommendationResponse(strategy="popular", items=_records(frame))

    @app.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
    def for_user(
        request: Request,
        user_id: int,
        k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
        preferred_genres: list[str] | None = Query(None),
    ):
        rec = get_model(request)
        try:
            frame = rec.recommend_for_user(user_id, k, preferred_genres)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        strategy = "hybrid" if rec.has_user(user_id) else "cold_start"
        return RecommendationResponse(strategy=strategy, items=_records(frame))

    @app.get("/recommendations/movie/{movie_id}", response_model=RecommendationResponse)
    def for_movie(
        request: Request,
        movie_id: int,
        k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
        mode: Literal["content", "collaborative"] = "content",
    ):
        rec = get_model(request)
        try:
            frame = rec.similar_movies(movie_id, k, mode)
        except UnknownMovieError:
            raise HTTPException(404, f"Unknown movie_id {movie_id}") from None
        return RecommendationResponse(strategy=f"similar_{mode}", items=_records(frame))

    return app


app = create_app()
