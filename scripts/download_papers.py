#!/usr/bin/env python3
"""
Parallel PDF downloader for academic papers from Google Scholar.
Uses Semantic Scholar, Unpaywall, and CrossRef APIs to find open-access PDFs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers" / "pdf"
METADATA_DIR = Path(__file__).resolve().parent.parent / "papers" / "metadata"
MAX_CONCURRENT = 8
RETRY_LIMIT = 2
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "ConnectomeLab-RAG/1.0 (mailto:connectome@snu.ac.kr)",
    "Accept": "application/pdf,application/json,*/*",
}


@dataclass
class PaperResult:
    title: str
    pdf_url: Optional[str] = None
    downloaded: bool = False
    filepath: Optional[str] = None
    error: Optional[str] = None
    source: Optional[str] = None


def sanitize_filename(title: str, max_len: int = 120) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', '', title)
    clean = re.sub(r'\s+', '_', clean.strip())
    if len(clean) > max_len:
        h = hashlib.md5(title.encode()).hexdigest()[:8]
        clean = clean[:max_len] + f"_{h}"
    return clean


async def search_semantic_scholar(
    session: aiohttp.ClientSession, title: str
) -> Optional[str]:
    """Query Semantic Scholar for open-access PDF."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 3, "fields": "title,openAccessPdf,externalIds"}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            if resp.status == 429:
                await asyncio.sleep(3)
                return None
            if resp.status != 200:
                return None
            data = await resp.json()
            for paper in data.get("data", []):
                oa = paper.get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
    except Exception:
        pass
    return None


async def search_unpaywall(
    session: aiohttp.ClientSession, title: str
) -> Optional[str]:
    """Query Unpaywall via CrossRef DOI lookup."""
    try:
        cr_url = "https://api.crossref.org/works"
        params = {"query.bibliographic": title, "rows": 3}
        async with session.get(cr_url, params=params, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            items = data.get("message", {}).get("items", [])
            for item in items:
                doi = item.get("DOI")
                if not doi:
                    continue
                up_url = f"https://api.unpaywall.org/v2/{doi}?email=connectome@snu.ac.kr"
                async with session.get(up_url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as up_resp:
                    if up_resp.status != 200:
                        continue
                    up_data = await up_resp.json()
                    best_oa = up_data.get("best_oa_location")
                    if best_oa:
                        pdf = best_oa.get("url_for_pdf") or best_oa.get("url")
                        if pdf:
                            return pdf
    except Exception:
        pass
    return None


async def search_pubmed_central(
    session: aiohttp.ClientSession, title: str
) -> Optional[str]:
    """Search PubMed Central for free full-text PDFs."""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pmc", "term": title, "retmax": 3, "retmode": "json"}
        async with session.get(search_url, params=params, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            for pmc_id in ids:
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
    except Exception:
        pass
    return None


async def download_pdf(
    session: aiohttp.ClientSession, url: str, filepath: Path
) -> bool:
    """Download a PDF file from a URL."""
    for attempt in range(RETRY_LIMIT):
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=60),
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    continue
                content_type = resp.headers.get("Content-Type", "")
                data = await resp.read()
                if len(data) < 1000:
                    return False
                if data[:5] == b"%PDF-" or "pdf" in content_type.lower():
                    filepath.write_bytes(data)
                    return True
                elif b"<html" in data[:500].lower():
                    return False
                else:
                    filepath.write_bytes(data)
                    return True
        except Exception:
            if attempt < RETRY_LIMIT - 1:
                await asyncio.sleep(1)
    return False


async def process_paper(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    paper: dict,
    idx: int,
    total: int,
) -> PaperResult:
    """Find and download PDF for a single paper."""
    title = paper["title"]
    result = PaperResult(title=title)

    async with sem:
        logger.info(f"[{idx+1}/{total}] Searching: {title[:70]}...")

        pdf_url = await search_semantic_scholar(session, title)
        if pdf_url:
            result.source = "SemanticScholar"
        
        if not pdf_url:
            await asyncio.sleep(0.5)
            pdf_url = await search_unpaywall(session, title)
            if pdf_url:
                result.source = "Unpaywall"

        if not pdf_url:
            await asyncio.sleep(0.5)
            pdf_url = await search_pubmed_central(session, title)
            if pdf_url:
                result.source = "PMC"

        if not pdf_url:
            result.error = "No open-access PDF found"
            logger.warning(f"[{idx+1}/{total}] No PDF: {title[:70]}")
            return result

        result.pdf_url = pdf_url
        filename = sanitize_filename(title) + ".pdf"
        filepath = PAPERS_DIR / filename

        if filepath.exists():
            result.downloaded = True
            result.filepath = str(filepath)
            logger.info(f"[{idx+1}/{total}] Already exists: {filename}")
            return result

        success = await download_pdf(session, pdf_url, filepath)
        if success:
            result.downloaded = True
            result.filepath = str(filepath)
            logger.info(f"[{idx+1}/{total}] Downloaded ({result.source}): {filename}")
        else:
            result.error = f"Download failed from {pdf_url}"
            logger.warning(f"[{idx+1}/{total}] Failed: {title[:70]}")

    return result


async def main() -> None:
    papers_json = Path(__file__).resolve().parent.parent.parent / "jiook_cha_scholar_papers.json"
    if not papers_json.exists():
        papers_json = Path("/home/juke/git/jiook_cha_scholar_papers.json")

    if not papers_json.exists():
        logger.error(f"Papers JSON not found: {papers_json}")
        sys.exit(1)

    papers = json.loads(papers_json.read_text())
    logger.info(f"Loaded {len(papers)} papers from {papers_json}")

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, force_close=True)

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        tasks = [
            process_paper(sem, session, paper, i, len(papers))
            for i, paper in enumerate(papers)
        ]
        results = await asyncio.gather(*tasks)

    downloaded = [r for r in results if r.downloaded]
    failed = [r for r in results if not r.downloaded]

    report = {
        "total": len(papers),
        "downloaded": len(downloaded),
        "failed": len(failed),
        "results": [
            {
                "title": r.title,
                "pdf_url": r.pdf_url,
                "downloaded": r.downloaded,
                "filepath": r.filepath,
                "source": r.source,
                "error": r.error,
            }
            for r in results
        ],
    }

    report_path = METADATA_DIR / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    logger.info("=" * 60)
    logger.info(f"DOWNLOAD COMPLETE: {len(downloaded)}/{len(papers)} papers")
    logger.info(f"Report saved to: {report_path}")
    if failed:
        logger.info(f"Failed papers ({len(failed)}):")
        for r in failed:
            logger.info(f"  - {r.title[:80]}: {r.error}")


if __name__ == "__main__":
    asyncio.run(main())
