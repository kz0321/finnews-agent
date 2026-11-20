"""Dataset validation and human-led evaluation, kept outside the agent inputs.

Verifier labels are deliberately never consumed by the scoring functions.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from .models import COMPANIES, Query, Source


def _dict(value: Any) -> dict:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


def load_dataset(path: str | Path) -> dict:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_dataset(dataset)
    if errors:
        raise ValueError("Invalid dataset:\n" + "\n".join(errors))
    return dataset


def validate_dataset(dataset: dict, strict_protocol: bool = False) -> list[str]:
    """Return actionable errors; strict mode checks the proposal's full protocol."""
    errors = []
    if dataset.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if dataset.get("dataset_kind") not in {"synthetic_demo", "real"}:
        errors.append("dataset_kind must be synthetic_demo or real")
    sources = dataset.get("sources", [])
    queries = dataset.get("queries", [])
    if not isinstance(sources, list) or not isinstance(queries, list):
        return errors + ["sources and queries must be lists"]
    source_map = {}
    for source in sources:
        sid = source.get("id")
        if not sid or sid in source_map:
            errors.append(f"Missing or duplicate source ID: {sid}")
        source_map[sid] = source
        if not source.get("text", "").strip():
            errors.append(f"Source {sid} has no original text")
        if source.get("published_at"):
            try:
                date.fromisoformat(source["published_at"])
            except (ValueError, TypeError):
                errors.append(f"Source {sid} has an invalid publication date")
    seen = set()
    windows = defaultdict(set)
    for query in queries:
        qid = query.get("id")
        if not qid or qid in seen:
            errors.append(f"Missing or duplicate query ID: {qid}")
        seen.add(qid)
        try:
            typed = Query.model_validate({key: query[key] for key in Query.model_fields if key in query})
        except Exception as exc:
            errors.append(f"Query {qid}: {exc}")
            continue
        if query.get("split") not in {"dev", "test"}:
            errors.append(f"Query {qid} needs split dev or test")
        if not typed.source_ids or len(typed.source_ids) != len(set(typed.source_ids)):
            errors.append(f"Query {qid} needs a nonempty, unique candidate pool")
        windows[typed.company].add((typed.as_of - timedelta(days=typed.window_days - 1), typed.as_of, query.get("split"), qid))
        for sid in typed.source_ids:
            if sid not in source_map:
                errors.append(f"Query {qid} references missing source {sid}")
            elif source_map[sid].get("company") != typed.company:
                errors.append(f"Query {qid} has cross-company source {sid}")
        facts = query.get("gold_facts", [])
        if strict_protocol and not 2 <= len(facts) <= 4:
            errors.append(f"Query {qid} must have 2-4 human-reviewed gold facts")
        fact_ids = set()
        for fact in facts:
            fid = fact.get("id")
            if not fid or fid in fact_ids:
                errors.append(f"Query {qid} has a missing or duplicate gold fact ID")
            fact_ids.add(fid)
            if not fact.get("evidence"):
                errors.append(f"Gold fact {fid} has no original evidence spans")
            for span in fact.get("evidence", []):
                sid, start, end = span.get("source_id"), span.get("start"), span.get("end")
                source = source_map.get(sid)
                if source is None or sid not in typed.source_ids:
                    errors.append(f"Gold fact {fid} references a source outside its pool")
                    continue
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(source.get("text", "")):
                    errors.append(f"Gold fact {fid} has invalid original-text offsets")
                publication = source.get("published_at")
                if not publication or publication > typed.as_of.isoformat():
                    errors.append(f"Gold fact {fid} must use known, pre-cutoff evidence")
        if strict_protocol and dataset.get("dataset_kind") == "real" and query.get("annotation_status") != "human_reviewed":
            errors.append(f"Query {qid} still needs human gold-span review")
    for company, periods in windows.items():
        development = [row for row in periods if row[2] == "dev"]
        test = [row for row in periods if row[2] == "test"]
        for dev_start, dev_end, _, dev_id in development:
            for test_start, test_end, _, test_id in test:
                if max(dev_start, test_start) <= min(dev_end, test_end):
                    errors.append(f"Overlapping development/test event windows for {company}: {dev_id}, {test_id}")
    if strict_protocol:
        if sum(q.get("split") == "dev" for q in queries) != 5 or sum(q.get("split") == "test" for q in queries) != 20:
            errors.append("Protocol requires exactly 5 development and 20 test queries")
        if {q.get("company") for q in queries} != set(COMPANIES):
            errors.append("Protocol requires the five configured companies")
        if not 30 <= len(sources) <= 50:
            errors.append("Protocol requires 30-50 source documents")
        if sum(bool(q.get("temporal_stress")) for q in queries if q.get("split") == "test") != 4:
            errors.append("Protocol requires exactly four temporal-stress test queries")
    return errors


def system_case(dataset: dict, query_id: str) -> tuple[Query, list[Source]]:
    """Allowlist agent inputs so references/labels cannot leak into prompts."""
    query_data = next((q for q in dataset["queries"] if q["id"] == query_id), None)
    if query_data is None:
        raise KeyError(query_id)
    query = Query.model_validate({key: query_data[key] for key in Query.model_fields if key in query_data})
    source_map = {source["id"]: source for source in dataset["sources"]}
    sources = [Source.model_validate({key: source_map[sid][key] for key in Source.model_fields if key in source_map[sid]}) for sid in query.source_ids]
    return query, sources


def _covered(start: int, end: int, intervals: Iterable[tuple[int, int]]) -> bool:
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def reference_coverage(gold_facts: list[dict], selected_chunks: Iterable[Any]) -> dict:
    """A gold fact counts only when every required span is covered by the union."""
    intervals = defaultdict(list)
    for chunk in selected_chunks:
        chunk = _dict(chunk)
        intervals[chunk["source_id"]].append((chunk["start"], chunk["end"]))
    covered = [fact["id"] for fact in gold_facts if fact.get("evidence") and all(
        _covered(span["start"], span["end"], intervals[span["source_id"]]) for span in fact["evidence"]
    )]
    return {"covered": len(covered), "total": len(gold_facts), "ratio": len(covered) / len(gold_facts) if gold_facts else None, "covered_fact_ids": covered}


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_blind_annotations(
    runs: Iterable[Any], outdir: str | Path, chunks: Iterable[Any] | None = None,
    seed: int = 17, double_score_fraction: float = 0.2,
) -> dict:
    """Export blinded claim sheets; mapping is separate and must be withheld.

    Split compound findings into additional unique claim_id rows before scoring.
    At least 20% of reports are independently assigned to a second annotator.
    """
    if not 0.2 <= double_score_fraction <= 1:
        raise ValueError("double_score_fraction must be between 0.2 and 1")
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    ordered = [_dict(run) for run in runs]
    random.Random(seed).shuffle(ordered)
    chunk_map = {_dict(chunk)["id"]: _dict(chunk) for chunk in (chunks or [])}
    mapping, rows, reports = [], [], []
    duplicate_count = math.ceil(len(ordered) * double_score_fraction)
    duplicate_ids = {f"R{index + 1:03d}" for index in range(duplicate_count)}
    for index, run in enumerate(ordered):
        blind_id = f"R{index + 1:03d}"
        query = run["query"]
        mapping.append({"blind_id": blind_id, "run_id": run["id"], "variant": run["variant"], "query_id": query["id"]})
        reports.append({"blind_id": blind_id, "company": query["company"], "as_of": str(query["as_of"]), "window_days": query["window_days"], "focus": query.get("focus", ""), "has_report": bool(run.get("brief")), "dataset_kind": run.get("dataset_kind", "unknown")})
        for finding in (run.get("brief") or {}).get("findings", []):
            evidence = []
            for cid in finding["evidence_ids"]:
                chunk = chunk_map.get(cid)
                evidence.append(f"[{cid}] {chunk.get('published_at') or 'DATE UNKNOWN'} | {chunk['url']} | {chunk['text']}" if chunk else f"[{cid}] EVIDENCE NOT SUPPLIED: resolve before annotation")
            rows.append({"blind_id": blind_id, "finding_id": finding["id"], "claim_id": finding["id"] + ".1", "claim_text": finding["text"], "evidence": "\n\n".join(evidence), "supported": "", "temporal_correct": "", "covered_gold_fact_ids": "", "annotator": "", "comment": ""})
    fields = ["blind_id", "finding_id", "claim_id", "claim_text", "evidence", "supported", "temporal_correct", "covered_gold_fact_ids", "annotator", "comment"]
    _write_csv(path / "primary_claims.csv", rows, fields)
    _write_csv(path / "secondary_claims.csv", [row for row in rows if row["blind_id"] in duplicate_ids], fields)
    _write_csv(path / "reports.csv", reports, ["blind_id", "company", "as_of", "window_days", "focus", "has_report", "dataset_kind"])
    (path / "PRIVATE_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    (path / "double_score_assignments.json").write_text(json.dumps(sorted(duplicate_ids), indent=2) + "\n", encoding="utf-8")
    return {"directory": str(path), "reports": len(ordered), "claims": len(rows), "double_scored_reports_requested": duplicate_count, "mapping": str(path / "PRIVATE_mapping.json")}


def read_annotation_csv(path: str | Path, mapping_path: str | Path, role: str = "primary") -> list[dict]:
    mapping = {row["blind_id"]: row["run_id"] for row in json.loads(Path(mapping_path).read_text(encoding="utf-8"))}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    annotations = []
    for row in rows:
        if not row.get("supported", "").strip() or not row.get("temporal_correct", "").strip():
            raise ValueError(f"Incomplete annotation row {row.get('blind_id')}/{row.get('claim_id')}; score every atomic claim before aggregation")
        if row.get("supported") not in {"0", "1"} or row.get("temporal_correct") not in {"0", "1", "unknown"}:
            raise ValueError("Annotations require supported 0/1 and temporal_correct 0/1/unknown")
        if row["blind_id"] not in mapping:
            raise ValueError(f"Unknown blind ID {row['blind_id']}")
        annotations.append({**row, "run_id": mapping[row["blind_id"]], "supported": int(row["supported"]), "temporal_correct": int(row["temporal_correct"]) if row["temporal_correct"] != "unknown" else "unknown", "covered_gold_fact_ids": [part.strip() for part in row.get("covered_gold_fact_ids", "").split(";") if part.strip()], "role": role})
    return annotations


def aggregate_human_scores(dataset: dict, runs: Iterable[Any], annotations: list[dict], split: str = "test") -> dict:
    """Score only explicit human labels; missing annotations remain unavailable.

    Every test query is in the completeness denominator. A missing/failed report
    contributes zero; an unannotated successful report prevents a final score.
    Claim and temporal ratios are micro averages on human-scored atomic claims.
    """
    queries = {q["id"]: q for q in dataset["queries"] if q["split"] == split}
    run_map = {}
    all_run_ids = set()
    for run in runs:
        run = _dict(run)
        if run["id"] in all_run_ids:
            raise ValueError(f"Duplicate run ID: {run['id']}")
        all_run_ids.add(run["id"])
        if run["query"]["id"] not in queries:
            continue
        key = (run["variant"], run["query"]["id"])
        if key in run_map:
            raise ValueError(f"Duplicate variant/query run: {key}; choose one frozen run")
        run_map[key] = run
    assessed_runs = {run["id"]: run for run in run_map.values()}
    by_run = defaultdict(dict)
    independent_primary = defaultdict(dict)
    secondary = defaultdict(dict)
    for annotation in annotations:
        if annotation["run_id"] not in all_run_ids:
            raise ValueError(f"Annotation references an unknown run ID: {annotation['run_id']}")
        if annotation["run_id"] not in assessed_runs:
            continue  # Development annotations never enter held-out statistics.
        if annotation.get("supported") not in {0, 1} or annotation.get("temporal_correct") not in {0, 1, "unknown"}:
            raise ValueError("Human labels must use supported 0/1 and temporal_correct 0/1/unknown")
        if not annotation.get("annotator"):
            raise ValueError("Human annotation requires a nonempty annotator identifier")
        target = secondary if annotation.get("role") == "secondary" else by_run
        key = annotation["claim_id"]
        old = target[annotation["run_id"]].get(key)
        if old and annotation.get("role") != "adjudicated":
            raise ValueError("Duplicate primary/secondary claim label; use explicit adjudication")
        target[annotation["run_id"]][key] = annotation
        if annotation.get("role", "primary") == "primary":
            independent_primary[annotation["run_id"]][key] = annotation
    variants = {}
    for variant in ("B0", "B1", "B2"):
        claims, completeness, task_rows, latencies = [], [], [], []
        unscored, valid, missing, failed, known_cost, unknown_cost = 0, 0, 0, 0, 0.0, 0
        for qid, query in queries.items():
            run = run_map.get((variant, qid))
            score = 0.0
            if run is None:
                missing += 1
            elif run.get("status") == "failed" or not run.get("brief"):
                failed += 1
            else:
                valid += 1
                labels = list(by_run.get(run["id"], {}).values())
                finding_ids = {finding["id"] for finding in run["brief"].get("findings", [])}
                label_finding_ids = {row["finding_id"] for row in labels}
                if not label_finding_ids.issubset(finding_ids):
                    raise ValueError(f"Annotation references a nonexistent finding in {run['id']}")
                if finding_ids - label_finding_ids:
                    unscored += 1
                    score = None
                else:
                    claims.extend(labels)
                    gold_ids = {fact["id"] for fact in query.get("gold_facts", [])}
                    matched = {fid for row in labels if row["supported"] == 1 and row["temporal_correct"] == 1 for fid in row.get("covered_gold_fact_ids", [])}
                    if matched - gold_ids:
                        raise ValueError(f"Annotation references an unknown gold fact in {qid}")
                    score = len(matched) / len(gold_ids) if gold_ids else None
                    if not gold_ids:
                        unscored += 1
            if score is not None:
                completeness.append(score)
            task_rows.append({"query_id": qid, "temporal_stress": bool(query.get("temporal_stress")), "key_fact_coverage": score})
            if run:
                duration = run.get("timings", {}).get("total_seconds")
                if isinstance(duration, (int, float)):
                    latencies.append(duration)
                for call in run.get("calls", []):
                    cost = call.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        known_cost += cost
                    else:
                        unknown_cost += 1
        variants[variant] = {
            "expected_queries": len(queries), "valid_reports": valid, "failed_reports": failed,
            "missing_runs": missing, "unscored_reports": unscored, "scored_atomic_claims": len(claims),
            "supported_and_cited_claim_ratio": sum(row["supported"] for row in claims) / len(claims) if claims else None,
            "temporal_correct_ratio": sum(row["temporal_correct"] == 1 for row in claims) / len(claims) if claims else None,
            "unknown_timing_claims": sum(row["temporal_correct"] == "unknown" for row in claims),
            "key_fact_coverage_all_queries": sum(completeness) / len(queries) if queries and not unscored else None,
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "known_llm_call_cost_usd": known_cost, "calls_with_unknown_cost": unknown_cost,
            "llm_call_cost_usd": known_cost if unknown_cost == 0 and missing == 0 else None,
            "total_cost_usd": None,
            "cost_scope": "LLM call records only; complete total unavailable without collection/search fee audit. Missing runs do not imply zero cost.",
            "paired_task_results": task_rows,
        }
    pairs = []
    double_runs = set()
    for run_id, labels in secondary.items():
        primary_labels = independent_primary.get(run_id, {})
        if set(labels) - set(primary_labels):
            raise ValueError("Secondary scoring references a claim without an independent primary label")
        for claim_id, other in labels.items():
            primary = primary_labels.get(claim_id)
            if primary:
                if primary["annotator"] == other["annotator"]:
                    raise ValueError("Double scoring requires two different annotators")
                pairs.append(primary["supported"] == other["supported"])
        run = assessed_runs[run_id]
        finding_ids = {row["id"] for row in (run.get("brief") or {}).get("findings", [])}
        primary_finding_ids = {row["finding_id"] for row in primary_labels.values()}
        if primary_labels and set(labels) == set(primary_labels) and primary_finding_ids == finding_ids and run.get("status") != "failed":
            double_runs.add(run_id)
    return {
        "dataset_kind": dataset["dataset_kind"], "split": split,
        "label_source": "human annotations only; verifier labels excluded",
        "variants": variants,
        "agreement": {"paired_atomic_claims": len(pairs), "raw_support_agreement": sum(pairs) / len(pairs) if pairs else None, "double_scored_reports": len(double_runs), "required_double_scored_reports": math.ceil(len(run_map) * 0.2)},
        "warning": "Synthetic fixture metrics are software checks, not evidence of financial research quality." if dataset["dataset_kind"] == "synthetic_demo" else "Results require reviewed source dates, gold spans and completed human scoring.",
    }


def retrieval_metrics(dataset: dict, runs: Iterable[Any], chunks_by_run: dict, split: str = "test") -> dict:
    """Paired reference coverage, including missing/failed attempts in denominator."""
    queries = {q["id"]: q for q in dataset["queries"] if q.get("split") == split}
    records = {}
    for run in runs:
        run = _dict(run)
        key = (run["variant"], run["query"]["id"])
        if key in records:
            raise ValueError("Duplicate variant/query run; choose one frozen run")
        records[key] = run
    variants = {}
    for variant in ("B0", "B1", "B2"):
        rows = []
        for query in queries.values():
            run = records.get((variant, query["id"]))
            row = {"query_id": query["id"], "temporal_stress": bool(query.get("temporal_stress")), "initial_coverage": None, "final_coverage": None}
            if query.get("gold_facts"):
                row.update(initial_coverage=0.0, final_coverage=0.0)
                if run and run.get("status") != "failed":
                    chunks = {_dict(c)["id"]: _dict(c) for c in chunks_by_run.get(run["id"], [])}
                    for field, ids_field in (("initial_coverage", "initial_selected_ids"), ("final_coverage", "selected_ids")):
                        ids = run.get(ids_field, [])
                        if any(cid not in chunks for cid in ids):
                            raise ValueError(f"Missing stored selected passage for run {run['id']}")
                        row[field] = reference_coverage(query["gold_facts"], [chunks[cid] for cid in ids])["ratio"]
            rows.append(row)
        scored = all(row["initial_coverage"] is not None for row in rows)
        variants[variant] = {"expected_queries": len(rows), "initial_coverage": statistics.mean(row["initial_coverage"] for row in rows) if rows and scored else None, "final_coverage": statistics.mean(row["final_coverage"] for row in rows) if rows and scored else None, "paired_task_results": rows}
    return {"dataset_kind": dataset["dataset_kind"], "split": split, "variants": variants, "is_human_quality_score": False, "note": "Reference-span coverage is retrieval coverage, not claim correctness. Unreviewed real spans cannot establish final experimental results."}


def main() -> int:
    """Standalone artifact scoring commands; no generation API calls are made."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    blind = sub.add_parser("export-blind")
    blind.add_argument("--runs", type=Path, required=True)
    blind.add_argument("--chunks", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)
    for command in ("score", "retrieval"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--dataset", type=Path, required=True)
        cmd.add_argument("--runs", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if command == "score":
            cmd.add_argument("--annotations", type=Path, required=True)
            cmd.add_argument("--mapping", type=Path, required=True)
            cmd.add_argument("--secondary", type=Path)
            cmd.add_argument("--adjudicated", type=Path)
        else:
            cmd.add_argument("--chunks", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text(encoding="utf-8"))
    if args.command == "export-blind":
        per_run = json.loads(args.chunks.read_text(encoding="utf-8"))
        result = export_blind_annotations(runs, args.output, [chunk for chunks in per_run.values() for chunk in chunks])
    else:
        dataset = load_dataset(args.dataset)
        if args.command == "retrieval":
            per_run = json.loads(args.chunks.read_text(encoding="utf-8"))
            result = retrieval_metrics(dataset, runs, per_run)
        else:
            annotations = read_annotation_csv(args.annotations, args.mapping)
            if args.secondary:
                annotations += read_annotation_csv(args.secondary, args.mapping, role="secondary")
            if args.adjudicated:
                annotations += read_annotation_csv(args.adjudicated, args.mapping, role="adjudicated")
            result = aggregate_human_scores(dataset, runs, annotations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
