## Semantic Patent Search Engine – High-Level Outline

### Goal
Provide a local, UI-accessible semantic search that:
- Accepts a patent document ID from the user.
- Scrapes the official patent page for title, abstract, and claims.
- Embeds the scraped text and searches a FAISS index of local patent data.
- Returns the top 5 most similar chunks with metadata and scores.

### Core Components
1. **Scraper Service (`app/scraper.py`)**
   - Requests patent pages (initial target: patents.google.com) using a custom User-Agent and timeout.
   - Parses the HTML for title, abstract, and claims using BeautifulSoup.
   - Normalizes whitespace and returns a structured result for downstream embedding.

2. **Local Data Loader (`app/data_loader.py`)**
   - Reads all JSON files under `UTF-8patent_data_small/data/patent_data_small`.
   - Splits each record into separate chunks: one for the abstract, one per claim.

3. **Embedding + Index Builder (`app/embeddings.py`, `app/index_builder.py`)**
   - Loads `sentence-transformers/all-MiniLM-L6-v2`.
   - Converts each abstract/claim chunk into a vector and builds **two** FAISS indices:
     - Abstract index (`abstracts.index` + metadata) for coarse document filtering.
     - Claim index (`claims.index` + metadata) for fine-grained chunk retrieval.

4. **Search API (`app/search.py`, `app/server.py`)**
   - On startup, loads both FAISS indices and their metadata catalogs.
   - When user submits a patent ID, the backend:
     1. Scrapes patent text (title + abstract + claims).
     2. Stage 1: embeds the scraped abstract and queries the abstract index to keep the top N document numbers.
     3. Stage 2: embeds the combined text and searches only the claims belonging to those N documents to find the most relevant chunks.
     4. Returns structured JSON for the UI (doc number, source type, snippet, similarity score).

5. **Web UI (`templates/index.html`, `static/styles.css`)**
   - Presents a form to enter an external patent ID.
   - Displays table of search results (document number, source type, snippet, similarity score).
   - Uses fetch/XHR to call the backend API without page reloads.

### Control Flow
1. **Startup**
   - Load settings (`app/config.py`).
   - Create/load both FAISS indices (build them first if absent).
2. **User Query**
   - UI sends the patent ID to `/api/search`.
   - Backend executes scrape → Stage 1 abstract filter → Stage 2 claim refinement.
3. **Response**
   - API returns the top semantic claim/abstract matches.
   - UI renders the table.
