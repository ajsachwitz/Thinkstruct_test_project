from __future__ import annotations

import os
from pathlib import Path
from typing import Final


class Settings:
    """Application-wide configuration with environment overrides."""

    DEFAULT_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        project_root = Path(
            os.getenv("PATENT_SEARCH_ROOT", Path(__file__).resolve().parents[1])
        )
        self.project_root: Path = project_root

        data_dir = os.getenv(
            "PATENT_SEARCH_DATA_DIR",
            project_root / "UTF-8patent_data_small" / "data" / "patent_data_small",
        )
        self.data_dir: Path = Path(data_dir)

        self.model_name: str = os.getenv(
            "PATENT_SEARCH_MODEL", self.DEFAULT_MODEL
        )
        self.index_top_k: int = int(os.getenv("PATENT_SEARCH_TOP_K", "10"))
        self.filter_top_k_docs: int = int(
            os.getenv("PATENT_SEARCH_FILTER_TOP_K", "100")
        )
        self.embedding_batch_size: int = int(
            os.getenv("PATENT_SEARCH_BATCH_SIZE", "64")
        )
        self.scraper_timeout_seconds: int = int(
            os.getenv("PATENT_SEARCH_SCRAPER_TIMEOUT", "20")
        )
        self.scraper_user_agent: str = os.getenv(
            "PATENT_SEARCH_USER_AGENT",
            "SemanticPatentSearchBot/1.0 (+https://thinkstruct.local)",
        )

        self.artifacts_dir: Path = Path(
            os.getenv("PATENT_SEARCH_ARTIFACTS_DIR", project_root / "artifacts")
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.abstract_index_path: Path = Path(
            os.getenv(
                "PATENT_SEARCH_ABSTRACT_INDEX_FILE",
                self.artifacts_dir / "abstracts.index",
            )
        )
        self.abstract_metadata_path: Path = Path(
            os.getenv(
                "PATENT_SEARCH_ABSTRACT_METADATA_FILE",
                self.artifacts_dir / "abstracts_metadata.json",
            )
        )
        self.claims_index_path: Path = Path(
            os.getenv(
                "PATENT_SEARCH_CLAIMS_INDEX_FILE",
                self.artifacts_dir / "claims.index",
            )
        )
        self.claims_metadata_path: Path = Path(
            os.getenv(
                "PATENT_SEARCH_CLAIMS_METADATA_FILE",
                self.artifacts_dir / "claims_metadata.json",
            )
        )
        self.doc_metadata_path: Path = Path(
            os.getenv(
                "PATENT_SEARCH_DOC_METADATA_FILE",
                self.artifacts_dir / "doc_metadata.json",
            )
        )


settings = Settings()
