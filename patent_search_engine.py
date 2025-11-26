"""
CLI entry point for running the Semantic Patent Search Engine FastAPI app.

Example:
    uvicorn patent_search_engine:app --reload
or:
    python patent_search_engine.py --reload
"""

from __future__ import annotations

import argparse

import uvicorn

from app.server import app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Semantic Patent Search Engine server."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "patent_search_engine:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
