## Semantic Patent Search Engine – Project Guide

### Overview
This project delivers a local-first semantic patent search experience. The workflow:
- Scrape a live patent page (title/abstract/claims) from patents.google.com using the ID entered in the UI.
- Embed the scraped **abstract** to filter for the most relevant local patents, then embed the combined title/abstract/claims text for final ranking.
- Compare those query vectors against two FAISS indices (abstracts + claims) built from the provided patent JSON corpus, where each abstract and claim is a separate searchable chunk.
- Show the top matches (document number, source type, snippet, similarity score) in a web UI.

### Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic.
- **Scraping & Parsing:** `requests`, `beautifulsoup4`.
- **Semantic Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Search:** FAISS (`IndexFlatIP` with cosine similarity via normalized embeddings).
- **Frontend:** Jinja2 template + vanilla JS + CSS in `templates/` and `static/`.
- **Data:** Local UTF-8 patent JSON files (one abstract + multiple claims per document) chunked via `app/data_loader.py`.

### Data Flow
1. **Index Build (offline/one-time)**
   - `app/data_loader.load_patent_chunks()` reads every JSON file and emits abstract/claim chunks.
   - `app.embeddings.embed_texts()` encodes each chunk, normalizes embeddings.
   - `app.index_builder.build_index_artifacts()` writes two FAISS indices + metadata files:
     - `artifacts/abstracts.index` / `abstracts_metadata.json`
     - `artifacts/claims.index` / `claims_metadata.json`

2. **Runtime Search (Two-Stage)**
   - FastAPI boots (`patent_search_engine:app`) and lazily loads both index pairs (`TwoStageSearchEngine`).
   - UI form submits `/api/search` with a patent ID.
   - `app.scraper.scrape_patent()` pulls the live title/abstract/claims.
   - Stage 1: embed the scraped abstract, query the abstracts index, keep the top-N document numbers (default 100).
   - Stage 2: embed the combined title+abstract+claims text, search only the claim vectors belonging to those documents, return the best chunks with cosine similarity scores.

### Setup & Running Locally
```powershell
cd C:\Users\AJSac\OneDrive\Desktop\Thinkstruct
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Build FAISS artifacts (first time or when data updates)
python -m app.index_builder --force

# Start the server
uvicorn patent_search_engine:app --reload
```

- Visit `http://localhost:8000`, enter a patent ID (e.g., `US20180368281A1`), and inspect the semantic matches.
- Logs display scraper and search info; errors return HTTP 400 (scraper issues) or 503/500 (runtime failures).

### Directory Highlights
- `app/`: Application modules (`scraper`, `data_loader`, `embeddings`, `index_builder`, `search`, `server`).
- `templates/`, `static/`: UI files.
- `docs/`: Architecture notes, file outlines, this README.
- `UTF-8patent_data_small/`: Source JSON corpus.
- `artifacts/`: Generated FAISS indices/metadata (`abstracts.*`, `claims.*`).
