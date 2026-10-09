"""FastAPI service for the Movie Recommendation System."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from recommender.config import ARTIFACT_PATH, DEFAULT_K, GENRE_COLS, MAX_K
from recommender.inference import Recommender, UnknownMovieError

logger = logging.getLogger("movie-recommender.api")

VERSION = "2.1.1"
GITHUB_URL = "https://github.com/johnthuo-analytics"
REPO_URL = "https://github.com/johnthuo-analytics/movie-recommender"

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"

SWAGGER_JS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"
SWAGGER_CSS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
SWAGGER_FAVICON = "https://fastapi.tiangolo.com/img/favicon.png"


# ---------------------------------------------------------------------------
# Documentation content
# ---------------------------------------------------------------------------

DESCRIPTION = """
A hybrid **movie recommendation service** built with **FastAPI** and trained on
the **MovieLens 100K** dataset.

## How it works

Recommendations blend four signals, tuned offline with a chronological
leave-last-out evaluation:

- **SVD** - matrix factorisation that learns latent user and movie tastes
- **Item-kNN** - movies that share similar rating patterns
- **Content** - genre similarity between movies
- **Popularity** - a robust fallback when a user has no history

Known users receive a **hybrid** ranking. Unknown users automatically receive
**cold-start** recommendations, optionally steered by preferred genres.

## Try it

- `GET /recommendations/user/196?k=5`
- `GET /recommendations/movie/50?mode=collaborative`
- `GET /recommendations/popular?preferred_genres=Comedy&preferred_genres=Drama`

## Response shape

Every recommendation endpoint returns the strategy that produced the result,
plus a ranked list of movies. For hybrid results each movie includes a
`components` object showing how much every signal contributed to its score.

```json
{
  "strategy": "hybrid",
  "items": [
    {
      "movie_id": 475,
      "title": "Trainspotting (1996)",
      "genres": ["Drama"],
      "score": 0.989,
      "components": {"content": 0.89, "knn": 0.0, "popularity": 0.0, "svd": 1.0}
    }
  ]
}
```

**Built by [John Thuo](https://github.com/johnthuo-analytics)** - released under the
MIT License.
"""

TAGS = [
    {
        "name": "Recommendations",
        "description": "Personalised, similar-movie and popularity-based recommendations.",
    },
    {
        "name": "Discovery",
        "description": "Catalogue information such as the available genres.",
    },
    {
        "name": "System",
        "description": "Service and model readiness.",
    },
]

DOCS_HERO = f"""
<header class="mr-hero">
  <div class="mr-hero__inner">
    <div>
      <div class="mr-hero__eyebrow">Hybrid recommender &bull; MovieLens 100K</div>
      <h1>🎬 Movie Recommendation API <span>v{VERSION}</span></h1>
      <p>Personalised picks, similar movies and genre-aware popularity,
         served from a pre-trained model.</p>
    </div>
    <nav class="mr-hero__links">
      <a href="/">Interactive demo</a>
      <a href="/health">Health</a>
      <a href="/openapi.json">OpenAPI JSON</a>
      <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>
    </nav>
  </div>
</header>
"""

DOCS_CSS = """
<style>
:root {
    --mr-bg: #f4f6fb;
    --mr-card: #ffffff;
    --mr-border: #e5e7eb;
    --mr-text: #111827;
    --mr-muted: #4b5563;
    --mr-accent: #7c5cff;
    --mr-accent-dark: #6a46f0;
    --mr-shadow: 0 6px 24px rgba(15, 23, 42, 0.07);
}

body {
    background: var(--mr-bg) !important;
    margin: 0;
}

/* ---- hero ---------------------------------------------------------- */
.mr-hero {
    background: linear-gradient(120deg, #0f172a 0%, #3b2a8f 55%, #7c5cff 100%);
    color: #fff;
    padding: 34px 20px 38px;
    margin-bottom: 26px;
    box-shadow: 0 10px 30px rgba(124, 92, 255, 0.28);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.mr-hero__inner {
    max-width: 1180px;
    margin: 0 auto;
    display: flex;
    flex-wrap: wrap;
    gap: 22px;
    align-items: center;
    justify-content: space-between;
}
.mr-hero__eyebrow {
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 8px;
}
.mr-hero h1 {
    margin: 0 0 8px;
    font-size: 32px;
    font-weight: 800;
    color: #fff;
}
.mr-hero h1 span {
    font-size: 13px;
    font-weight: 700;
    vertical-align: middle;
    background: rgba(255, 255, 255, 0.16);
    border-radius: 999px;
    padding: 4px 11px;
    margin-left: 6px;
}
.mr-hero p {
    margin: 0;
    max-width: 560px;
    line-height: 1.6;
    opacity: 0.88;
}
.mr-hero__links {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.mr-hero__links a {
    color: #fff !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    padding: 9px 16px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.28);
    transition: background 0.15s ease, transform 0.15s ease;
}
.mr-hero__links a:hover {
    background: rgba(255, 255, 255, 0.24);
    transform: translateY(-1px);
}

/* ---- swagger shell ------------------------------------------------- */
.swagger-ui {
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 18px 40px;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.swagger-ui .topbar { display: none; }
.swagger-ui .information-container { margin: 0 0 24px; }

.swagger-ui .info {
    background: var(--mr-card);
    padding: 26px 32px;
    border-radius: 16px;
    border: 1px solid var(--mr-border);
    box-shadow: var(--mr-shadow);
    margin: 0;
}
/* The hero already shows the title, so hide Swagger's own copy. */
.swagger-ui .info .title { display: none; }
.swagger-ui .info p,
.swagger-ui .info li { color: var(--mr-muted); line-height: 1.7; }
.swagger-ui .info h1,
.swagger-ui .info h2,
.swagger-ui .info h3 { color: var(--mr-text); }

.swagger-ui .markdown h2 {
    font-size: 21px;
    margin-top: 26px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--mr-border);
}
.swagger-ui .markdown code {
    background: #eef2ff;
    color: var(--mr-accent-dark);
    padding: 2px 6px;
    border-radius: 5px;
}
.swagger-ui .markdown pre {
    background: #0f172a;
    border-radius: 12px;
    padding: 16px;
    overflow-x: auto;
}
.swagger-ui .markdown pre code {
    background: transparent;
    color: #e5e7eb;
    padding: 0;
}

/* ---- sections & operations ---------------------------------------- */
.swagger-ui .opblock-tag {
    color: var(--mr-text);
    font-size: 20px;
    font-weight: 700;
    border-bottom: 1px solid var(--mr-border);
    padding: 16px 8px;
}
.swagger-ui .opblock {
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    overflow: hidden;
    margin: 0 0 14px;
    transition: box-shadow 0.15s ease;
}
.swagger-ui .opblock:hover { box-shadow: 0 10px 26px rgba(15, 23, 42, 0.11); }
.swagger-ui .opblock .opblock-summary-method {
    border-radius: 8px;
    font-weight: 800;
    min-width: 76px;
}
.swagger-ui .opblock .opblock-summary-description {
    color: #374151;
    font-weight: 600;
}

.swagger-ui .btn.execute {
    background: var(--mr-accent) !important;
    border-color: var(--mr-accent) !important;
    border-radius: 8px;
    font-weight: 700;
}
.swagger-ui .btn.execute:hover {
    background: var(--mr-accent-dark) !important;
    border-color: var(--mr-accent-dark) !important;
}
.swagger-ui .btn { border-radius: 8px; }

.swagger-ui .responses-inner { border-radius: 10px; }
.swagger-ui table thead tr th { background: #f8fafc; color: #374151; }
.swagger-ui section.models {
    border: 1px solid var(--mr-border);
    border-radius: 12px;
    background: var(--mr-card);
    box-shadow: var(--mr-shadow);
}
.swagger-ui .scheme-container {
    background: var(--mr-card);
    border-radius: 12px;
    border: 1px solid var(--mr-border);
    box-shadow: var(--mr-shadow);
}
.swagger-ui a { color: var(--mr-accent); }
.swagger-ui a:hover { color: #5b3fd6; }

.swagger-ui::after {
    content: "Movie Recommendation API \\2022  Built by John Thuo";
    display: block;
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    margin: 34px 0 10px;
}

@media (max-width: 640px) {
    .mr-hero h1 { font-size: 24px; }
    .swagger-ui .info { padding: 20px; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Recommendation(BaseModel):
    """A single recommended movie."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "movie_id": 475,
                "title": "Trainspotting (1996)",
                "genres": ["Drama"],
                "score": 0.989,
                "components": {"content": 0.89, "knn": 0.0, "popularity": 0.0, "svd": 1.0},
            }
        }
    )

    movie_id: int = Field(description="MovieLens movie ID.")
    title: str = Field(description="Movie title, including release year.")
    genres: list[str] = Field(description="Genres assigned to the movie.")
    score: float = Field(description="Ranking score. Higher is better.")
    components: dict[str, float] | None = Field(
        default=None,
        description="Per-signal contribution to the score (hybrid results only).",
    )


class RecommendationResponse(BaseModel):
    """A ranked list of recommendations and the strategy that produced it."""

    strategy: Literal[
        "hybrid",
        "cold_start",
        "similar_content",
        "similar_collaborative",
        "popular",
    ] = Field(description="How the recommendations were generated.")
    items: list[Recommendation] = Field(description="Movies, best match first.")


class GenresResponse(BaseModel):
    """Genres known to the recommender."""

    count: int = Field(description="Number of genres.")
    genres: list[str] = Field(description="Genre names accepted by `preferred_genres`.")


class HealthResponse(BaseModel):
    """Service readiness."""

    status: Literal["ok", "starting"] = Field(description="`ok` once the model is loaded.")
    service: str
    version: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Standard error body."""

    detail: str = Field(description="Human-readable explanation.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _records(frame) -> list[Recommendation]:
    """Convert recommendation records into API objects."""
    return [Recommendation(**row) for row in frame.to_dict(orient="records")]


def _normalise_genres(preferred_genres: list[str] | None) -> list[str] | None:
    """Validate requested genres (case-insensitive).

    Raises a 400 listing the valid genres if any requested genre is unknown.
    """
    if not preferred_genres:
        return None

    available = {str(genre).strip().lower(): str(genre).strip() for genre in GENRE_COLS}
    cleaned: list[str] = []
    unknown: list[str] = []

    for genre in preferred_genres:
        key = str(genre).strip().lower()
        if key in available:
            cleaned.append(available[key])
        else:
            unknown.append(str(genre))

    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown genre(s): {', '.join(unknown)}. "
                f"Valid genres: {', '.join(available.values())}."
            ),
        )
    return cleaned or None


def get_model(request: Request) -> Recommender:
    """Dependency that returns the loaded model or a 503 if it is not ready."""
    recommender = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation model is not ready.")
    return recommender


def _operation_id(route: APIRoute) -> str:
    """Use the function name as the operation ID.

    Gives short, stable Swagger deep links, e.g.
    ``/docs#/Recommendations/popular_recommendations``.
    """
    return route.name


ModelDep = Annotated[Recommender, Depends(get_model)]

KParam = Annotated[
    int,
    Query(ge=1, le=MAX_K, description=f"Number of movies to return (1-{MAX_K})."),
]
GenresParam = Annotated[
    list[str] | None,
    Query(
        description="Optional preferred genres. Repeat the parameter for several genres.",
        examples=["Drama"],
    ),
]
ModeParam = Annotated[
    Literal["content", "collaborative"],
    Query(description="`content` compares genres; `collaborative` compares rating patterns."),
]

BAD_GENRE = {
    400: {"model": ErrorResponse, "description": "One or more genres are not recognised."},
}

NOT_READY = {
    503: {
        "model": ErrorResponse,
        "description": "The model artifact has not been loaded yet.",
    }
}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    model: Recommender | None = None,
    model_path: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        model: An already-loaded recommender (handy for tests). When given,
            nothing is read from disk.
        model_path: Artifact to load at startup. Falls back to the
            ``MODEL_PATH`` environment variable, then the default artifact path.
    """
    configured_path = Path(model_path or os.getenv("MODEL_PATH", str(ARTIFACT_PATH)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model is not None:
            app.state.recommender = model
        else:
            try:
                app.state.recommender = Recommender.load(configured_path)
                logger.info("Loaded recommender from %s", configured_path)
            except Exception:
                logger.exception("Could not load recommender from %s", configured_path)
                app.state.recommender = None
        yield

    app = FastAPI(
        title="🎬 Movie Recommendation API",
        summary="Hybrid movie recommendations on MovieLens 100K.",
        description=DESCRIPTION,
        version=VERSION,
        openapi_version="3.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_tags=TAGS,
        contact={"name": "John Thuo", "url": GITHUB_URL},
        license_info={"name": "MIT License", "url": f"{REPO_URL}/blob/main/LICENSE"},
        swagger_ui_parameters={
            "docExpansion": "list",
            "defaultModelsExpandDepth": 0,
            "defaultModelExpandDepth": 2,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "syntaxHighlight.theme": "arta",
        },
        generate_unique_id_function=_operation_id,
        lifespan=lifespan,
    )

    # -- middleware ---------------------------------------------------------

    cors_origins = os.getenv("CORS_ORIGINS", "*").strip()
    allowed_origins = (
        ["*"] if cors_origins == "*" else [o.strip() for o in cors_origins.split(",") if o.strip()]
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-ms"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag every response with a request ID and its processing time."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        return response

    # -- pages --------------------------------------------------------------

    @app.get("/docs", include_in_schema=False)
    def custom_swagger_ui():
        response = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title="Movie Recommendation API \u2022 Swagger",
            swagger_js_url=SWAGGER_JS,
            swagger_css_url=SWAGGER_CSS,
            swagger_favicon_url=SWAGGER_FAVICON,
            # docs_url=None bypasses FastAPI's built-in docs route, which is
            # what normally applies these settings, so pass them explicitly.
            swagger_ui_parameters=app.swagger_ui_parameters,
        )
        html = (
            response.body.decode("utf-8")
            .replace("</head>", DOCS_CSS + "</head>", 1)
            .replace('<div id="swagger-ui">', DOCS_HERO + '<div id="swagger-ui">', 1)
        )
        # A new response so Content-Length is recomputed.
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False)
    def home():
        if not INDEX_HTML.exists():
            raise HTTPException(status_code=404, detail="Interactive demo is not available.")
        return FileResponse(INDEX_HTML)

    # -- system -------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Check API and model status",
        description="Returns `ok` once the model artifact is loaded, `starting` before that.",
    )
    def health(request: Request):
        loaded = getattr(request.app.state, "recommender", None) is not None
        return {
            "status": "ok" if loaded else "starting",
            "service": "movie-recommendation-api",
            "version": VERSION,
            "model_loaded": loaded,
        }

    # -- discovery ----------------------------------------------------------

    @app.get(
        "/genres",
        response_model=GenresResponse,
        tags=["Discovery"],
        summary="List available movie genres",
        description="Genre names you can pass as `preferred_genres` (case-insensitive).",
    )
    def genres():
        return {"count": len(GENRE_COLS), "genres": list(GENRE_COLS)}

    # -- recommendations ----------------------------------------------------

    @app.get(
        "/recommendations/popular",
        response_model=RecommendationResponse,
        tags=["Recommendations"],
        summary="Get popular movie recommendations",
        description=(
            "Popularity-ranked movies, optionally limited to the given genres. "
            "Also the fallback used for users with no history."
        ),
        responses={**NOT_READY, **BAD_GENRE},
    )
    def popular_recommendations(
        recommender: ModelDep,
        k: KParam = DEFAULT_K,
        preferred_genres: GenresParam = None,
    ):
        frame = recommender.popular(k=k, preferred_genres=_normalise_genres(preferred_genres))
        return RecommendationResponse(strategy="popular", items=_records(frame))

    @app.get(
        "/recommendations/user/{user_id}",
        response_model=RecommendationResponse,
        tags=["Recommendations"],
        summary="Get personalised recommendations",
        description=(
            "Hybrid recommendations for a MovieLens user, excluding movies they have "
            "already rated. Unknown users get `cold_start` popularity results instead "
            "of an error."
        ),
        responses={**NOT_READY, **BAD_GENRE},
    )
    def user_recommendations(
        user_id: Annotated[int, PathParam(description="MovieLens user ID.", examples=[196])],
        recommender: ModelDep,
        k: KParam = DEFAULT_K,
        preferred_genres: GenresParam = None,
    ):
        selected = _normalise_genres(preferred_genres)

        if recommender.has_user(user_id):
            frame = recommender.recommend_for_user(
                user_id=user_id, k=k, preferred_genres=selected
            )
            strategy = "hybrid"
        else:
            frame = recommender.popular(k=k, preferred_genres=selected)
            strategy = "cold_start"

        return RecommendationResponse(strategy=strategy, items=_records(frame))

    @app.get(
        "/recommendations/movie/{movie_id}",
        response_model=RecommendationResponse,
        tags=["Recommendations"],
        summary="Find similar movies",
        description=(
            "Movies similar to the selected one, using genre similarity (`content`) "
            "or shared rating patterns (`collaborative`)."
        ),
        responses={
            **NOT_READY,
            404: {"model": ErrorResponse, "description": "The movie ID is not in the catalogue."},
        },
    )
    def movie_recommendations(
        movie_id: Annotated[int, PathParam(description="MovieLens movie ID.", examples=[50])],
        recommender: ModelDep,
        k: KParam = DEFAULT_K,
        mode: ModeParam = "content",
    ):
        try:
            frame = recommender.similar_movies(movie_id=movie_id, k=k, mode=mode)
        except UnknownMovieError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        strategy = "similar_content" if mode == "content" else "similar_collaborative"
        return RecommendationResponse(strategy=strategy, items=_records(frame))

    return app


app = create_app()
