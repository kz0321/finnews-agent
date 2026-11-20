"""Fetch permitted official HTML into a private, ignored snapshot for annotation.

No API key is required. Dates are seed metadata requiring human confirmation.
403/429, redirects off the allowlist, and unreadable pages are recorded as failures;
this script does not bypass access restrictions or replace sources with snippets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
HOSTS = {"www.apple.com", "www.microsoft.com", "nvidianews.nvidia.com", "ir.aboutamazon.com", "abc.xyz"}


def fetch_html(client, url: str) -> tuple[str, str]:
    for _ in range(4):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in HOSTS:
            raise ValueError("URL/redirect is outside the fixed official-source allowlist")
        with client.stream("GET", url) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            if "html" not in response.headers.get("content-type", "").lower():
                raise ValueError("Only readable HTML is supported by this course downloader")
            payload = bytearray()
            for block in response.iter_bytes():
                payload.extend(block)
                if len(payload) > 8_000_000:
                    raise ValueError("HTML exceeds the 8 MB source limit")
            return payload.decode(response.encoding or "utf-8", errors="replace"), url
    raise ValueError("Too many redirects")


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # ASP.NET investor-relations sites wrap the entire article inside one form.
    # Removing all form elements would silently delete the actual disclosure.
    for element in soup.select("script, style, nav, header, footer, noscript, svg"):
        element.decompose()
    # Prefer a publisher's article body. Preserve table row/column delimiters so
    # annotators can detect table-extraction errors instead of trusting flat numbers.
    selectors = [".article-body", ".article-content", ".news-release", ".module_news-detail", "article", "main", "#mainContent"]
    main = next((node for selector in selectors if (node := soup.select_one(selector)) and len(node.get_text()) > 600), soup.body or soup)
    for row in list(main.find_all("tr")):
        cells = row.find_all(["th", "td"], recursive=False)
        if cells:
            row.replace_with("\n" + " | ".join(cell.get_text(" ", strip=True) for cell in cells) + "\n")
    text = main.get_text("\n", strip=True)
    text = "\n".join(re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip())
    if len(text) < 600:
        raise ValueError("Insufficient original body text; manual source review needed")
    return text


def candidate_spans(source: dict) -> list[dict]:
    """Suggest inspectable spans only. These are NOT gold annotations."""
    output = []
    for match in re.finditer(r"[^\n]+", source["text"]):
        passage = match.group().strip()
        if 60 <= len(passage) <= 750 and re.search(r"revenue|sales|operating (?:income|profit)|guidance|outlook|risk", passage, re.I) and re.search(r"\d", passage):
            output.append({"source_id": source["id"], "start": match.start(), "end": match.end(), "text": passage, "status": "needs_human_review_not_gold"})
        if len(output) == 6:
            break
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "data/evaluation/real_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/private/real-corpus")
    parser.add_argument("--limit", type=int, default=None, help="Optional development smoke subset, never a complete benchmark")
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--resume", action="store_true", help="Reuse verified downloaded originals in this output directory")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    sources, log, seen_hashes = [], [], set()
    cached = {}
    prior_dataset = args.output / "dataset.json"
    if args.resume and prior_dataset.exists():
        previous = json.loads(prior_dataset.read_text(encoding="utf-8"))
        for source in previous.get("sources", []):
            digest = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
            if source.get("content_hash") != digest:
                raise ValueError(f"Existing snapshot hash mismatch: {source['id']}")
            cached[source["id"]] = source
            seen_hashes.add(digest)
    seeds = plan["source_seeds"][:args.limit] if args.limit else plan["source_seeds"]
    with httpx.Client(timeout=30, follow_redirects=False, headers={"User-Agent": "Mozilla/5.0 (compatible; FinNewsCourseProject/0.1; academic research)"}) as client:
        for seed in seeds:
            stamp = datetime.now(timezone.utc).isoformat()
            if seed["id"] in cached and cached[seed["id"]]["url"] == seed["url"]:
                source = cached[seed["id"]]
                sources.append(source)
                log.append({"id": source["id"], "url": source["url"], "status": "cached_existing_snapshot_needs_date_and_content_review", "content_hash": source["content_hash"], "characters": len(source["text"]), "retrieved_at": source["retrieved_at"]})
                print(f"CACHED {source['id']}", flush=True)
                continue
            try:
                html, final_url = fetch_html(client, seed["url"])
                text = extract_text(html)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    raise ValueError("Duplicate normalized original text; do not inflate document count")
                seen_hashes.add(digest)
                source = {key: seed[key] for key in ("id", "company", "title", "published_at")}
                source.update(url=final_url, text=text, retrieved_at=stamp, content_hash=digest)
                sources.append(source)
                log.append({"id": seed["id"], "url": final_url, "status": "downloaded_needs_date_and_content_review", "content_hash": digest, "characters": len(text), "retrieved_at": stamp})
                print(f"OK {seed['id']}: {len(text)} characters", flush=True)
            except Exception as exc:
                log.append({"id": seed["id"], "url": seed["url"], "status": "failed", "error": f"{type(exc).__name__}: {exc}", "retrieved_at": stamp})
                print(f"FAILED {seed['id']}: {type(exc).__name__}", flush=True)
            time.sleep(max(0.5, args.delay))
    available = {source["id"] for source in sources}
    queries = []
    for query in plan["queries"]:
        if query["primary_source_id"] not in available:
            continue
        row = {**query, "source_ids": [sid for sid in query["source_ids"] if sid in available]}
        row["annotation_candidates"] = [span for source in sources if source["id"] == query["primary_source_id"] for span in candidate_spans(source)]
        queries.append(row)
    dataset = {"schema_version": 1, "dataset_kind": "real", "status": "unreviewed_snapshot_not_a_completed_benchmark", "description": "Private downloaded originals. Seed dates and candidate spans require human review; gold_facts intentionally empty.", "sources": sources, "queries": queries}
    (args.output / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"requested_documents": len(seeds), "downloaded_documents": len(sources), "prepared_queries": len(queries), "human_reviewed_queries": 0, "gold_facts": 0, "downloads": log}
    (args.output / "download-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(sources)} private originals and {len(queries)} unreviewed queries to {args.output}")
    print("Do not publish full source texts. Complete human dates/spans before evaluation.")
    return int(any(row["status"] == "failed" for row in log))


if __name__ == "__main__":
    raise SystemExit(main())
