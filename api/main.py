"""
AI Movie Recommendation Engine — FastAPI backend
Connects to Google Gemini (free tier: gemini-2.5-flash-lite / gemini-1.5-pro)
and returns live, personalized movie recommendations based on user preferences.
"""

import json
import logging
import os
import re
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie-recommender")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. Set it in a .env file or environment "
        "variable before calling /api/recommend."
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="AI Movie Recommendation Engine",
    description="Takes user preferences and returns live, personalized movie "
    "recommendations powered by Google Gemini.",
    version="1.0.0",
)

# Allow the frontend (served from anywhere during dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RecommendationRequest(BaseModel):
    genres: List[str] = Field(default_factory=list, description="Preferred genres")
    mood: Optional[str] = Field(None, description="Mood, e.g. 'feel-good', 'intense'")
    favorite_movies: Optional[str] = Field(
        None, description="Comma-separated movies/shows the user already likes"
    )
    era: Optional[str] = Field(
        None, description="Preferred era, e.g. 'classic', '90s', 'recent'"
    )
    language: Optional[str] = Field(None, description="Preferred film language")
    avoid: Optional[str] = Field(None, description="Genres/themes to avoid")
    count: int = Field(6, ge=1, le=12, description="Number of recommendations")


class MovieRecommendation(BaseModel):
    title: str
    year: Optional[str] = None
    genre: Optional[str] = None
    reason: str
    rating_hint: Optional[str] = None


class RecommendationResponse(BaseModel):
    model_used: str
    recommendations: List[MovieRecommendation]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_prompt(prefs: RecommendationRequest) -> str:
    parts = [
        "You are a knowledgeable film critic and recommendation engine.",
        "Recommend real, existing movies that match the user's preferences below.",
        "Do not repeat any title the user already listed as a favorite.",
    ]

    if prefs.genres:
        parts.append(f"Preferred genres: {', '.join(prefs.genres)}.")
    if prefs.mood:
        parts.append(f"Desired mood/vibe: {prefs.mood}.")
    if prefs.favorite_movies:
        parts.append(f"Movies the user already loves: {prefs.favorite_movies}.")
    if prefs.era:
        parts.append(f"Preferred era: {prefs.era}.")
    if prefs.language:
        parts.append(f"Preferred film language/origin: {prefs.language}.")
    if prefs.avoid:
        parts.append(f"Avoid these genres/themes: {prefs.avoid}.")

    parts.append(
        f"Return exactly {prefs.count} recommendations as a JSON array. "
        'Each item must be an object with keys: "title" (string), '
        '"year" (string), "genre" (string), "reason" (a 1-2 sentence '
        "explanation of why this fits the user's taste), and "
        '"rating_hint" (a short phrase like "Widely acclaimed" or '
        '"Cult favorite", not a numeric score). '
        "Return ONLY the JSON array, no markdown fences, no extra text."
    )
    return "\n".join(parts)


def extract_json_array(text: str) -> list:
    """Gemini is asked for pure JSON, but this strips markdown fences etc.
    just in case, as a defensive fallback."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model response")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
_INDEX_HTML_PATH = os.path.join(_PUBLIC_DIR, "index.html")


@app.get("/", include_in_schema=False)
def serve_index():
    if os.path.isfile(_INDEX_HTML_PATH):
        return FileResponse(_INDEX_HTML_PATH)
    raise HTTPException(status_code=404, detail=f"index.html not found at {_INDEX_HTML_PATH}")


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend_movies(prefs: RecommendationRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: GEMINI_API_KEY is not set.",
        )

    if not prefs.genres and not prefs.mood and not prefs.favorite_movies:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: genres, mood, or favorite_movies.",
        )

    prompt = build_prompt(prefs)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.8,
            },
        )
        raw_text = response.text
    except Exception as exc:  # network/auth/quota errors from the Gemini SDK
        logger.exception("Gemini API call failed")
        raise HTTPException(
            status_code=502, detail=f"Gemini API request failed: {exc}"
        ) from exc

    try:
        items = extract_json_array(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse Gemini response: %s", raw_text)
        raise HTTPException(
            status_code=502,
            detail="Gemini returned a response that could not be parsed as JSON.",
        ) from exc

    recommendations = [
        MovieRecommendation(
            title=item.get("title", "Unknown title"),
            year=item.get("year"),
            genre=item.get("genre"),
            reason=item.get("reason", ""),
            rating_hint=item.get("rating_hint"),
        )
        for item in items
    ]

    return RecommendationResponse(
        model_used=GEMINI_MODEL, recommendations=recommendations
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
