from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)

GOOGLE_PATENT_URL = "https://patents.google.com/patent/{patent_id}/en?oq={patent_id}"


class PatentScraperError(RuntimeError):
    """Raised when we cannot retrieve or parse the external patent page."""


@dataclass(frozen=True)
class PatentScrapeResult:
    patent_id: str
    title: str
    abstract: str
    claims: List[str]


def scrape_patent(patent_id: str) -> PatentScrapeResult:
    """
    Scrape title, abstract, and claims from patents.google.com for the given patent ID.

    Raises:
        PatentScraperError: if the page cannot be retrieved or parsed.
    """
    normalized_id = patent_id.strip()
    if not normalized_id:
        raise PatentScraperError("Patent ID cannot be empty.")

    url = GOOGLE_PATENT_URL.format(patent_id=normalized_id)
    logger.info("Scraping patent page: %s", url)

    try:
        resp = requests.get(
            url,
            timeout=settings.scraper_timeout_seconds,
            headers={"User-Agent": settings.scraper_user_agent},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise PatentScraperError(f"Failed to fetch patent page: {exc}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")

    title = _extract_title(soup) or ""
    abstract = _extract_abstract(soup) or ""
    claims = _extract_claims(soup)

    if not any([title, abstract, claims]):
        raise PatentScraperError(
            f"Unable to parse patent content for ID '{normalized_id}'."
        )

    logger.info(
        "Scraped patent %s: title len=%s, abstract len=%s, claims=%s",
        normalized_id,
        len(title),
        len(abstract),
        len(claims),
    )

    return PatentScrapeResult(patent_id=normalized_id, title=title, abstract=abstract, claims=claims)


def _extract_title(soup: BeautifulSoup) -> str:
    for selector in [
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "DC.title"}),
        ("meta", {"itemprop": "name"}),
    ]:
        meta = soup.find(*selector)
        if meta and meta.get("content"):
            return meta["content"].strip()

    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    return ""


def _extract_abstract(soup: BeautifulSoup) -> str:
    abstract_container = soup.find(attrs={"itemprop": "abstract"})
    if abstract_container:
        return abstract_container.get_text(" ", strip=True)

    meta = soup.find("meta", {"name": "DC.description"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return ""


def _extract_claims(soup: BeautifulSoup) -> List[str]:
    claims_section = soup.find(attrs={"itemprop": "claims"})
    if not claims_section:
        return []

    claim_texts: List[str] = []
    for claim in claims_section.find_all(attrs={"itemprop": "claimText"}):
        text = claim.get_text(" ", strip=True)
        if text:
            claim_texts.append(text)

    if not claim_texts:
        # Fallback: look for elements with class "claim-text".
        for claim in claims_section.select(".claim-text"):
            text = claim.get_text(" ", strip=True)
            if text:
                claim_texts.append(text)

    return claim_texts
