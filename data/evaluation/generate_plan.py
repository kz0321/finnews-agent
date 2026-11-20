"""Generate metadata-only real-company evaluation seeds. Never invent gold labels."""
import json
from datetime import date, timedelta
from pathlib import Path


def build_plan():
    sources, queries = [], []
    apple = [("2024-02-01", "2024/02", "first"), ("2024-05-02", "2024/05", "second"), ("2024-08-01", "2024/08", "third"), ("2024-10-31", "2024/10", "fourth"), ("2025-01-30", "2025/01", "first")]
    microsoft = [("2024-01-30", "2024-Q2"), ("2024-04-25", "2024-Q3"), ("2024-07-30", "2024-Q4"), ("2024-10-30", "2025-Q1"), ("2025-01-29", "2025-Q2")]
    nvidia = [("2024-02-21", "fourth-quarter-and-fiscal-2024"), ("2024-05-22", "first-quarter-fiscal-2025"), ("2024-08-28", "second-quarter-fiscal-2025"), ("2024-11-20", "third-quarter-fiscal-2025"), ("2025-02-26", "fourth-quarter-and-fiscal-2025")]
    amazon = [("2024-02-01", "2024", "Fourth-Quarter-Results"), ("2024-04-30", "2024", "First-Quarter-Results-68b9258cd"), ("2024-08-01", "2024", "Second-Quarter-Results"), ("2024-10-31", "2024", "Third-Quarter-Results"), ("2025-02-06", "2025", "Fourth-Quarter-Results")]
    alphabet = [("2024-01-30", "2024/2023-q4-earnings-call"), ("2024-04-25", "2024/2024-q1-earnings-call"), ("2024-07-23", "2024/2024-Q2-Earnings-Call"), ("2024-10-29", "2024/2024-Q3-Earnings-Call"), ("2025-02-04", "2025/2024-Q4-Earnings-Call")]
    company_urls = {
        "AAPL": [(d, f"https://www.apple.com/newsroom/{part}/apple-reports-{quarter}-quarter-results/") for d, part, quarter in apple],
        "MSFT": [(d, f"https://www.microsoft.com/en-us/Investor/earnings/FY-{period}/press-release-webcast") for d, period in microsoft],
        "NVDA": [(d, f"https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-{period}") for d, period in nvidia],
        "AMZN": [(d, f"https://ir.aboutamazon.com/news-release/news-release-details/{year}/Amazon-com-Announces-{period}/default.aspx") for d, year, period in amazon],
        "GOOGL": [(d, f"https://abc.xyz/investor/events/event-details/{part}/") for d, part in alphabet],
    }
    for ticker, entries in company_urls.items():
        for i, (published, url) in enumerate(entries):
            source_id = f"real-{ticker.lower()}-{published}"
            sources.append({"id": source_id, "company": ticker, "url": url, "title": f"{ticker} official earnings {'call' if ticker == 'GOOGL' else 'release'} {published}", "published_at": published, "date_basis": "Dated official release/event; must be reviewed against downloaded original", "status": "seed_not_collected"})
            pool = [source_id]
            if i > 0:
                pool.append(f"real-{ticker.lower()}-{entries[i - 1][0]}")
            if ticker == "MSFT":
                extra_id = source_id + "-performance"
                sources.append({"id": extra_id, "company": ticker, "url": url.replace("press-release-webcast", "performance"), "title": f"Microsoft performance and outlook {published}", "published_at": published, "date_basis": "Associated fiscal-quarter earnings page; review before evaluation", "status": "seed_not_collected"})
                pool.append(extra_id)
            stress = i == 1 and ticker != "GOOGL"
            if stress:
                pool.append(f"real-{ticker.lower()}-{entries[i + 1][0]}")
            focus = [
                "Summarize reported revenue, operating performance and disclosed business risks; distinguish the reporting period from the release date.",
                "Explain the reported revenue movement and management outlook, retaining units and differentiating forecasts from realized results.",
                "Summarize key business-segment performance and one material announcement; distinguish quarterly from year-to-date figures.",
                "Summarize profitability, cash returns or investment priorities, and any explicitly disclosed uncertainty.",
                "Summarize the most recent earnings and guidance, distinguishing annual totals from quarterly amounts and dated background.",
            ][i]
            queries.append({"id": f"real-{ticker.lower()}-{'dev' if i == 0 else 'test'}-{i + 1}", "split": "dev" if i == 0 else "test", "company": ticker, "as_of": (date.fromisoformat(published) + timedelta(days=2)).isoformat(), "window_days": 7 if i % 2 == 0 else 30, "focus": focus, "source_ids": pool, "primary_source_id": source_id, "gold_facts": [], "annotation_status": "needs_human_review", "temporal_stress": stress})
    return {"schema_version": 1, "dataset_kind": "real", "status": "metadata_only_not_an_evaluated_corpus", "description": "25 prespecified real-company research windows, 30 official HTML source seeds. Download originals privately, verify dates and annotate 2-4 gold facts before generation. No reference answers are supplied.", "source_seeds": sources, "queries": queries}


if __name__ == "__main__":
    output = Path(__file__).with_name("real_plan.json")
    output.write_text(json.dumps(build_plan(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
