"""FastAPI inference layer. Serves a pre-trained artifact; never trains."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from recommender.config import ARTIFACT_PATH, DEFAULT_K, GENRE_COLS, MAX_K
from recommender.inference import Recommender, UnknownMovieError

VERSION = "2.1.0"
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
DESCRIPTION = (
    "Hybrid movie recommender on **MovieLens 100K**: item-kNN + SVD + genre content + "
    "popularity, tuned on a chronological validation split.\n\n"
    "Open the [interactive demo](/) or try the endpoints below."
)
DOCS_CSS = """
<style>
body{background:#f5f6fb;font-family:system-ui,'Segoe UI',sans-serif}
.swagger-ui .info{margin:36px 0 18px}
.swagger-ui .info .title{font-size:38px;letter-spacing:-.02em;
  background:linear-gradient(90deg,#7c5cff,#22d3ee);-webkit-background-clip:text;color:transparent}
.swagger-ui .info .title small{background:#7c5cff}
.swagger-ui .opblock-tag{border-bottom:1px solid #e3e6f2;font-size:20px}
.swagger-ui .opblock{border-radius:12px;border:1px solid #e3e6f2;background:#fff;
  box-shadow:0 2px 8px #7c5cff12;margin:0 0 12px}
.swagger-ui .opblock.opblock-get .opblock-summary-method{background:#7c5cff;border-radius:8px}
.swagger-ui .opblock.opblock-get{border-color:#d9d2ff}
.swagger-ui .btn.execute{background:#7c5cff;border-color:#7c5cff;border-radius:8px}
.swagger-ui .btn.try-out__btn{border-color:#7c5cff;color:#7c5cff;border-radius:8px}
.swagger-ui .scheme-container{box-shadow:none;background:transparent}
</style>
"""
TAGS = [
    {"name": "recommendations", "description": "Personalised, similar-movie and popular lists."},
    {"name": "system", "description": "Service status."},
]


EXAMPLE = {
    "strategy": "hybrid",
    "items": [
        {
            "movie_id": 475,
            "title": "Trainspotting (1996)",
            "genres": ["Drama"],
            "score": 0.989,
            "components": {"knn": 0.0, "svd": 1.0, "content": 0.89, "popularity": 0.0},
        }
    ],
}


class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float
    components: dict[str, float] | None = None


class RecommendationResponse(BaseModel):
    strategy: Literal["hybrid", "cold_start", "similar_content", "similar_collaborative", "popular"]
    items: list[Recommendation]

    model_config = {"json_schema_extra": {"example": EXAMPLE}}


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

    app = FastAPI(
        title="Movie Recommendation API",
        version=VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
        docs_url=None,
        contact={"name": "John Thuo", "url": "https://github.com/johnthuo-analytics"},
        license_info={"name": "MIT"},
    )

    @app.get("/docs", include_in_schema=False)
    def docs():
        page = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title="Movie Recommender · API docs",
            swagger_ui_parameters={
                "defaultModelsExpandDepth": -1,
                "docExpansion": "list",
                "displayRequestDuration": True,
                "tryItOutEnabled": True,
                "syntaxHighlight.theme": "obsidian",
            },
        )
        html = page.body.decode().replace("</head>", DOCS_CSS + "</head>")
        return HTMLResponse(html)

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"]
    )

    @app.get("/", include_in_schema=False)
    def demo():
        return FileResponse(INDEX_HTML)

    @app.get("/genres", tags=["system"], summary="List valid genres")
    def genres():
        return {"genres": list(GENRE_COLS)}

    def get_model(request: Request) -> Recommender:
        loaded = request.app.state.model
        if loaded is None:
            raise HTTPException(503, "Model artifact not loaded. Run `python train.py` first.")
        return loaded

    @app.get("/health", tags=["system"], summary="Service and model status")
    def health(request: Request):
        return {
            "status": "ok",
            "version": VERSION,
            "model_loaded": request.app.state.model is not None,
        }

    @app.get(
        "/recommendations/popular",
        response_model=RecommendationResponse,
        tags=["recommendations"],
        summary="Popular movies (cold start)",
        description="Popularity ranking, optionally boosted by `preferred_genres`.",
    )
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

    @app.get(
        "/recommendations/user/{user_id}",
        response_model=RecommendationResponse,
        tags=["recommendations"],
        summary="Personalised recommendations",
        description=(
            "Hybrid ranking (kNN + SVD + content + popularity) with per-component scores. "
            "Unknown users fall back to cold start."
        ),
    )
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

    @app.get(
        "/recommendations/movie/{movie_id}",
        response_model=RecommendationResponse,
        tags=["recommendations"],
        summary="Movies similar to a given movie",
        description="`mode=content` uses genres; `mode=collaborative` uses viewers' taste.",
    )
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
