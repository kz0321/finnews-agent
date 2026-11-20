"""Regenerate the explicitly fictional, self-authored classroom evaluation fixture.

These are NOT historical company disclosures and NOT a financial benchmark.
No web content or reference answers enter generation through this script.
"""

from __future__ import annotations

import json
from pathlib import Path


COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
}


def build() -> dict:
    sources, queries = [], []
    for company_index, (ticker, company) in enumerate(COMPANIES.items(), start=1):
        background_id = f"demo-{ticker.lower()}-background"
        unknown_id = f"demo-{ticker.lower()}-undated"
        background = (
            f"Synthetic classroom scenario for {ticker}; all figures are fictional. "
            f"On June 15, 2025, the scenario's {company} board approved an annual "
            f"research budget of {company_index + 3} million US dollars. "
            "This background decision predates the later monthly operating updates."
        )
        sources.append(make_source(background_id, ticker, f"Fictional {company} background", background, "2025-06-15"))
        for month in range(7, 12):
            source_id = f"demo-{ticker.lower()}-2025-{month:02d}"
            # Obvious toy values intentionally avoid resemblance to real results.
            revenue = 100 + company_index * 10 + month
            previous = revenue - 8
            profit = 10 + company_index + month
            sentences = [
                f"In the fictional {company} scenario, revenue for the month ended 2025-{month:02d}-09 was {revenue} million US dollars, compared with {previous} million US dollars in the prior month.",
                f"Operating profit for that same fictional monthly period was {profit} million US dollars.",
                f"The scenario update, published on 2025-{month:02d}-10, stated that a supplier delay could affect next month's deliveries.",
            ]
            text = f"Synthetic classroom scenario for {ticker}; all figures and events are fictional. " + " ".join(sentences)
            source = make_source(source_id, ticker, f"Fictional {company} monthly update {month:02d}/2025", text, f"2025-{month:02d}-10")
            sources.append(source)
            gold = []
            for fact_number, sentence in enumerate(sentences, start=1):
                start = text.index(sentence)
                gold.append({
                    "id": f"{source_id}-fact-{fact_number}",
                    "text": sentence,
                    "evidence": [{"source_id": source_id, "start": start, "end": start + len(sentence)}],
                })
            query_id = f"{ticker.lower()}-{'dev' if month == 7 else 'test'}-{month:02d}"
            source_ids = [source_id, background_id, unknown_id]
            # Four test questions deliberately include a dated future distractor.
            temporal_stress = month == 8 and company_index <= 4
            if temporal_stress:
                source_ids.append(f"demo-{ticker.lower()}-2025-09")
            queries.append({
                "id": query_id,
                "split": "dev" if month == 7 else "test",
                "company": ticker,
                "as_of": f"2025-{month:02d}-12",
                "window_days": 7 if month % 2 else 30,
                "focus": "Summarize the fictional monthly revenue, operating profit, and disclosed delivery risk; retain units and reporting periods.",
                "source_ids": source_ids,
                "gold_facts": gold,
                "temporal_stress": temporal_stress,
            })
        unknown = (
            f"Synthetic classroom scenario for {ticker}; this passage is fictional and undated. "
            f"An undated note claims that {company} might increase deliveries. "
            "The note does not establish when an announcement or event occurred."
        )
        sources.append(make_source(unknown_id, ticker, f"Fictional undated {company} note", unknown, None))
    return {
        "schema_version": 1,
        "dataset_kind": "synthetic_demo",
        "description": "Self-authored fictional fixtures for software acceptance. Not real market evidence, not the proposed human-scored financial benchmark.",
        "license": "MIT (self-authored text)",
        "sources": sources,
        "queries": queries,
    }


def make_source(source_id: str, company: str, title: str, text: str, published_at: str | None) -> dict:
    return {
        "id": source_id,
        "company": company,
        "url": f"https://example.com/finnews-synthetic/{source_id}",
        "title": title,
        "text": text,
        "published_at": published_at,
        "retrieved_at": "2026-11-20T00:00:00+00:00",
        "license": "MIT (self-authored fictional fixture)",
    }


if __name__ == "__main__":
    output = Path(__file__).with_name("dataset.json")
    output.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote fictional fixture dataset: {output}")
