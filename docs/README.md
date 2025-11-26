# Semantic Patent Search Engine (Thinkstruct Project)

This project implements a high-precision, two-stage semantic search engine for U.S. patent data, designed for prior art discovery. It uses Python, FAISS, and state-of-the-art embedding models, exposed through a FastAPI backend and lightweight HTML/JS frontend.

---

## 1. Problem Statement

**Goal**
Given an external patent document ID as a query, efficiently identify and rank the top 10 most relevant patents from a local directory of patents.

---

## 2. Solution and Enhancement Strategy

The system evolves from a naive whole-document embedding approach to a high-precision, hybrid semantic–lexical pipeline.

### Part 1: Naive Semantic Search (Single-Vector Indexing)

**Initial approach**
- Concatenate Title + Abstract + all Claims into one long string per patent.
- Embed this entire text as one vector and store it in a FAISS index.
- Query by scraping the patent, concatenating the sections, embedding once, and searching nearest neighbors.

**Limitation:** A single vector for a 10k–20k word patent causes semantic dilution. Fine-grained meaning in individual claims is lost.
### Part 2: Enhancements

### Enhancement 1: Two-Stage Semantic Search

**Granular Indexing**
- **Abstract Index:** one vector per patent abstract.
- **Claim Index:** one vector per individual claim across all patents.

**Two-Stage Retrieval Pipeline**
1. **Stage 1 — Fast Abstract Search**
   - Embed the scraped abstract as vector \(Q\).
   - Search the Abstract Index.
   - Retrieve the top 100 candidate patents.
2. **Stage 2 — Claim-to-Claim Precision Refinement**
   - For each candidate patent, compare each query claim against each candidate claim.
   - Record the best single claim-pair similarity score.
   - Rank candidates using this maximum match; return the top 10 unique patents.

**Outcome:** Final results reflect the most semantically similar individual claim, providing dramatically improved prior-art accuracy.

### Enhancement 2: Hybrid Keyword Filtering

Some workflows require mandatory keyword constraints (e.g., a term must appear in the Title or Abstract).

**Mechanism**
- Optional `keyword_filter` field in the UI.
- After Stage 1, filter the top 100 candidates by string matching in Title/Abstract.
- Only patents that pass this lexical test enter the expensive Stage 2 refinement.

**Result:** Final results are both semantically relevant and lexically compliant.

---

## 3. How to Run the Code

This project is intended for local execution using a Python environment and Uvicorn.

### Prerequisites
- Python 3.8+
- FAISS CPU (`faiss-cpu`)
- SentenceTransformers (`sentence-transformers`)

### Step 1: Clone the Repository & Set Up the Environment
```bash
git clone <your_repo_link>
cd patent_search_engine
python -m venv .venv
```

Activate the environment:
- **Windows:** `.\.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data & Build FAISS Indices
- Place your `patents_ipa{DATE}.json` file(s) into the `data/` directory.
- Then run:
```bash
python -m app.index_builder --force
```
This embeds abstracts and claims and generates:
- `abstracts.index`
- `claims.index`

### Step 3: Run the FastAPI Server
```bash
uvicorn patent_search_engine:app --reload ^
    --reload-dir app ^
    --reload-dir templates ^
    --reload-dir static ^
    --reload-dir docs
```

### Step 4: Open the Web UI
Navigate to `http://127.0.0.1:8000/` and enter:
- External patent ID (e.g., `US10945783B2`)
- Optional keyword filter

Then run a full hybrid, two-stage semantic search.

---

## Summary of Features
- **Two-stage semantic retrieval**
  - Abstract filter → fast shortlist
  - Claim-to-claim refinement → precision match
- **Granular FAISS indices**
  - Abstract index + Claim index
- **Hybrid keyword filtering**
  - Ensures lexical + semantic relevance
- **FastAPI backend + simple HTML/JS UI**
  - Easy to operate locally
- **High-precision prior-art ranking**
  - Outputs the most similar claim per patent and the matching query claim

---

## Directory Highlights
- `app/` – Scraper, data loader, embeddings, FAISS builder, search engine, server.
- `templates/`, `static/` – Web UI.
- `docs/` – Architecture notes (this README + `architecture.md`).
- `UTF-8patent_data_small/` – Local JSON corpus.
- `artifacts/` – Generated FAISS indices/metadata (`abstracts.*`, `claims.*`).

Rebuild the FAISS artifacts whenever the corpus or embedding settings change, then restart Uvicorn. The UI will show the scraped patent details and the top 10 prior-art patents, each referenced by the claim that best matches the query patent’s claims.
