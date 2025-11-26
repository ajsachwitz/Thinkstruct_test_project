from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.scraper import PatentScraperError
from app.search import run_semantic_search, serialize_search_response

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Semantic Patent Search Engine",
    description="Local semantic patent search prototype with scraper verification.",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


class SearchRequest(BaseModel):
    patent_id: str = Field(..., min_length=4, max_length=64, description="Patent document identifier")
    keyword_filter: str | None = Field(
        default=None,
        description="Single keyword or phrase that must appear in candidate titles/abstracts.",
        max_length=200,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/search")
async def api_search(payload: SearchRequest) -> dict:
    try:
        response = run_semantic_search(payload.patent_id, keyword_filter=payload.keyword_filter)
    except PatentScraperError as exc:
        logger.warning("Scraper error for %s: %s", payload.patent_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Search error for %s: %s", payload.patent_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unhandled error for %s", payload.patent_id)
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc

    return serialize_search_response(response)
