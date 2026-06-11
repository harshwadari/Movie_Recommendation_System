import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD ENV
# =========================
load_dotenv()
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_IMG_BACKDROP = "https://image.tmdb.org/t/p/w1280"

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Recommendation API",
    description="TF-IDF + TMDB powered movie recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "Models"

DF_PATH = MODELS_DIR / "df.pkl"
INDICES_PATH = MODELS_DIR / "indices.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# HELPER: BUILD TITLE → IDX MAP
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Build a normalized title → integer index map from the loaded indices object.
    Supports pandas Series (title → idx) or dict shapes.
    """
    title_map: Dict[str, int] = {}

    if isinstance(indices, pd.Series):
        # Series where index = title, value = positional index
        for title, idx in indices.items():
            title_map[str(title).strip().lower()] = int(idx)
    elif isinstance(indices, dict):
        for title, idx in indices.items():
            title_map[str(title).strip().lower()] = int(idx)
    else:
        # Fallback: build from df itself
        if df is not None and "title" in df.columns:
            for i, row_title in enumerate(df["title"]):
                title_map[str(row_title).strip().lower()] = i

    return title_map


# =========================
# HELPER: GET LOCAL IDX BY TITLE
# =========================
def get_local_idx_by_title(query_title: str) -> int:
    """
    Return the positional index (row number in df) for the given title.
    Raises HTTPException 404 if not found.
    """
    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="Title index map not loaded")

    key = query_title.strip().lower()
    idx = TITLE_TO_IDX.get(key)

    if idx is None:
        # Fuzzy fallback: check if query is a substring of any title
        for stored_title, stored_idx in TITLE_TO_IDX.items():
            if key in stored_title:
                idx = stored_idx
                break

    if idx is None:
        raise HTTPException(
            status_code=404, detail=f"Movie not found in local dataset: '{query_title}'"
        )

    return idx


# =========================
# HELPER: TMDB HTTP GET
# =========================
async def tmdb_get(path: str, params: Optional[Dict] = None) -> Optional[dict]:
    """Perform a TMDB API GET request. Returns parsed JSON or None on error."""
    if not TMDB_API_KEY:
        return None
    base_params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    if params:
        base_params.update(params)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=base_params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    print("\n========== LOADING MODELS ==========")

    required_files = [
        DF_PATH,
        INDICES_PATH,
        TFIDF_MATRIX_PATH,
        TFIDF_PATH,
    ]

    for file in required_files:
        if not file.exists():
            raise RuntimeError(f"Missing file: {file}")

    print(f"DF_PATH: {DF_PATH}")
    print(f"INDICES_PATH: {INDICES_PATH}")
    print(f"TFIDF_MATRIX_PATH: {TFIDF_MATRIX_PATH}")
    print(f"TFIDF_PATH: {TFIDF_PATH}")

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("df.pkl does not contain a pandas DataFrame")

    if "title" not in df.columns:
        raise RuntimeError("DataFrame must contain a 'title' column")

    print(f"Movies Loaded: {len(df)}")
    print("Models Loaded Successfully")
    print("====================================\n")


# =========================
# ROOT ROUTE
# =========================
@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API",
        "status": "running",
        "docs": "/docs"
    }


# =========================
# TF-IDF RECOMMENDER
# =========================
def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources not loaded"
        )

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]

    scores = cosine_similarity(
        qv,
        tfidf_matrix
    ).flatten()

    order = np.argsort(-scores)

    recommendations: List[Tuple[str, float]] = []

    for i in order:
        if int(i) == int(idx):
            continue

        try:
            title = str(df.iloc[int(i)]["title"])
        except Exception:
            continue

        recommendations.append(
            (
                title,
                float(scores[int(i)])
            )
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# =========================
# ROUTE: /tmdb/search
# =========================
@app.get("/tmdb/search")
async def tmdb_search(query: str = Query(..., min_length=1)):
    """Search movies by keyword via TMDB."""
    data = await tmdb_get("/search/movie", {"query": query, "page": 1})
    if data is None:
        raise HTTPException(status_code=502, detail="TMDB search failed")
    return data


# =========================
# ROUTE: /home
# =========================
CATEGORY_PATHS: Dict[str, str] = {
    "trending": "/trending/movie/week",
    "popular": "/movie/popular",
    "top_rated": "/movie/top_rated",
    "now_playing": "/movie/now_playing",
    "upcoming": "/movie/upcoming",
}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home_feed(
    category: str = Query("trending"),
    limit: int = Query(24, ge=1, le=100),
):
    """Return a list of movie cards for the home feed."""
    path = CATEGORY_PATHS.get(category)
    if path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Choose from: {list(CATEGORY_PATHS.keys())}"
        )

    data = await tmdb_get(path)
    if data is None:
        raise HTTPException(status_code=502, detail="TMDB request failed")

    results = data.get("results") or []
    cards: List[TMDBMovieCard] = []
    for m in results[:limit]:
        tmdb_id = m.get("id")
        title = m.get("title") or m.get("name") or ""
        if not tmdb_id or not title:
            continue
        poster_path = m.get("poster_path")
        cards.append(
            TMDBMovieCard(
                tmdb_id=int(tmdb_id),
                title=title,
                poster_url=f"{TMDB_IMG}{poster_path}" if poster_path else None,
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return cards


# =========================
# ROUTE: /movie/id/{tmdb_id}
# =========================
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_by_id(tmdb_id: int):
    """Fetch full movie details from TMDB by TMDB ID."""
    data = await tmdb_get(f"/movie/{tmdb_id}")
    if data is None:
        raise HTTPException(status_code=404, detail=f"Movie {tmdb_id} not found on TMDB")

    poster_path = data.get("poster_path")
    backdrop_path = data.get("backdrop_path")

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=f"{TMDB_IMG}{poster_path}" if poster_path else None,
        backdrop_url=f"{TMDB_IMG_BACKDROP}{backdrop_path}" if backdrop_path else None,
        genres=data.get("genres") or [],
    )


# =========================
# ROUTE: /movie/search  (SEARCH BUNDLE)
# =========================
@app.get("/movie/search", response_model=SearchBundleResponse)
async def movie_search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(10, ge=1, le=50),
    genre_limit: int = Query(12, ge=1, le=50),
):
    """
    Main recommendation bundle:
    1. Find the movie on TMDB
    2. TF-IDF similar movies from local dataset
    3. Genre-based TMDB recommendations
    """
    # 1. TMDB search for query
    tmdb_search_data = await tmdb_get("/search/movie", {"query": query, "page": 1})
    if not tmdb_search_data or not tmdb_search_data.get("results"):
        raise HTTPException(status_code=404, detail=f"No TMDB results for '{query}'")

    # Pick the best matching result
    results = tmdb_search_data["results"]
    best = None
    query_lower = query.strip().lower()
    for r in results:
        if (r.get("title") or "").strip().lower() == query_lower:
            best = r
            break
    if best is None:
        best = results[0]

    tmdb_id = int(best["id"])
    title = best.get("title") or query
    poster_path = best.get("poster_path")
    backdrop_path = best.get("backdrop_path")

    # Fetch full details for details model
    detail_data = await tmdb_get(f"/movie/{tmdb_id}")
    if detail_data:
        movie_details = TMDBMovieDetails(
            tmdb_id=tmdb_id,
            title=detail_data.get("title") or title,
            overview=detail_data.get("overview"),
            release_date=detail_data.get("release_date"),
            poster_url=f"{TMDB_IMG}{detail_data.get('poster_path')}" if detail_data.get("poster_path") else None,
            backdrop_url=f"{TMDB_IMG_BACKDROP}{detail_data.get('backdrop_path')}" if detail_data.get("backdrop_path") else None,
            genres=detail_data.get("genres") or [],
        )
    else:
        movie_details = TMDBMovieDetails(
            tmdb_id=tmdb_id,
            title=title,
            poster_url=f"{TMDB_IMG}{poster_path}" if poster_path else None,
        )

    # 2. TF-IDF recommendations
    tfidf_recs: List[TFIDFRecItem] = []
    try:
        raw_recs = tfidf_recommend_titles(title, top_n=tfidf_top_n)
        for rec_title, rec_score in raw_recs:
            # Try to find TMDB card for each rec title
            tmdb_card: Optional[TMDBMovieCard] = None
            tmdb_data = await tmdb_get("/search/movie", {"query": rec_title, "page": 1})
            if tmdb_data and tmdb_data.get("results"):
                m = tmdb_data["results"][0]
                m_poster = m.get("poster_path")
                tmdb_card = TMDBMovieCard(
                    tmdb_id=int(m["id"]),
                    title=m.get("title") or rec_title,
                    poster_url=f"{TMDB_IMG}{m_poster}" if m_poster else None,
                    release_date=m.get("release_date"),
                    vote_average=m.get("vote_average"),
                )
            tfidf_recs.append(
                TFIDFRecItem(title=rec_title, score=rec_score, tmdb=tmdb_card)
            )
    except HTTPException:
        # Movie not in local dataset — TF-IDF fallback is empty
        pass

    # 3. Genre-based recommendations from TMDB
    genre_recs: List[TMDBMovieCard] = []
    genre_data = await tmdb_get(f"/movie/{tmdb_id}/recommendations", {"page": 1})
    if genre_data and genre_data.get("results"):
        for m in genre_data["results"][:genre_limit]:
            m_id = m.get("id")
            m_title = m.get("title") or ""
            if not m_id or not m_title:
                continue
            m_poster = m.get("poster_path")
            genre_recs.append(
                TMDBMovieCard(
                    tmdb_id=int(m_id),
                    title=m_title,
                    poster_url=f"{TMDB_IMG}{m_poster}" if m_poster else None,
                    release_date=m.get("release_date"),
                    vote_average=m.get("vote_average"),
                )
            )

    return SearchBundleResponse(
        query=query,
        movie_details=movie_details,
        tfidf_recommendations=tfidf_recs,
        genre_recommendations=genre_recs,
    )


# =========================
# ROUTE: /recommend/genre
# =========================
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_by_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """Return TMDB genre/similar recommendations for a movie by TMDB ID."""
    data = await tmdb_get(f"/movie/{tmdb_id}/recommendations", {"page": 1})
    if data is None:
        raise HTTPException(status_code=502, detail="TMDB request failed")

    results = data.get("results") or []
    cards: List[TMDBMovieCard] = []
    for m in results[:limit]:
        m_id = m.get("id")
        m_title = m.get("title") or ""
        if not m_id or not m_title:
            continue
        m_poster = m.get("poster_path")
        cards.append(
            TMDBMovieCard(
                tmdb_id=int(m_id),
                title=m_title,
                poster_url=f"{TMDB_IMG}{m_poster}" if m_poster else None,
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return cards