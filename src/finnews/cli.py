"""Commands for repeatable research and blinded evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from .models import COMPANIES, Query
from .service import DEMO_DATASET, data_dir, load_corpus, research


def parser():
    p = argparse.ArgumentParser(prog="finnews", description="Evidence-grounded financial news research")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "batch"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--dataset", type=Path, default=DEMO_DATASET if name != "run" else None,
                         required=name == "run")
        cmd.add_argument("--output", type=Path, default=None)
        if name != "batch":
            cmd.add_argument("--query-id", default=None)
            cmd.add_argument("--variant", choices=["B0", "B1", "B2"], default="B2")
        else:
            cmd.add_argument("--split", choices=["dev", "test", "all"], default="test")
        if name != "demo":
            cmd.add_argument("--demo", action="store_true", help="Synthetic fixtures and hash encoder only")
    collect = sub.add_parser("collect")
    collect.add_argument("--company", choices=COMPANIES, required=True)
    collect.add_argument("--as-of", type=date.fromisoformat, required=True)
    collect.add_argument("--days", type=int, choices=[7, 30], default=30)
    collect.add_argument("--focus", default="earnings, material announcements and risks")
    collect.add_argument("--output", type=Path, default=None)
    validate = sub.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--strict", action="store_true")
    blind = sub.add_parser("blind", help="Export variant-hidden annotation sheets")
    blind.add_argument("--runs", type=Path, required=True)
    blind.add_argument("--chunks", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)
    coverage = sub.add_parser("coverage", help="Measure original-span reference coverage")
    coverage.add_argument("--dataset", type=Path, required=True)
    coverage.add_argument("--runs", type=Path, required=True)
    coverage.add_argument("--chunks", type=Path, required=True)
    coverage.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score", help="Aggregate completed human annotation sheets")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--runs", type=Path, required=True)
    score.add_argument("--primary", type=Path, required=True)
    score.add_argument("--secondary", type=Path)
    score.add_argument("--mapping", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return p


async def execute(args):
    default_root = data_dir()  # Always load .env, including when --output overrides the path.
    root = args.output if hasattr(args, "output") and args.output else default_root
    if args.command in {"blind", "coverage", "score"}:
        from .evaluation import (aggregate_human_scores, export_blind_annotations,
                                 load_dataset, read_annotation_csv, reference_coverage)
        runs = json.loads(args.runs.read_text(encoding="utf-8"))
        if args.command == "blind":
            chunk_groups = json.loads(args.chunks.read_text(encoding="utf-8"))
            chunks = {c["id"]: c for group in chunk_groups.values() for c in group}
            result = export_blind_annotations(runs, root, chunks=list(chunks.values()))
        elif args.command == "coverage":
            dataset = load_dataset(args.dataset)
            queries = {q["id"]: q for q in dataset["queries"]}
            groups = json.loads(args.chunks.read_text(encoding="utf-8"))
            result = {"dataset_kind": dataset["dataset_kind"], "metric": "reference-evidence span coverage", "runs": []}
            for run in runs:
                available = {c["id"]: c for c in groups[run["id"]]}
                facts = queries[run["query"]["id"]].get("gold_facts", [])
                result["runs"].append({"run_id": run["id"], "query_id": run["query"]["id"],
                    "variant": run["variant"], "initial": reference_coverage(facts,
                        [available[i] for i in run["initial_selected_ids"]]),
                    "final": reference_coverage(facts, [available[i] for i in run["selected_ids"]])})
            root.parent.mkdir(parents=True, exist_ok=True)
            root.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            dataset = load_dataset(args.dataset)
            annotations = read_annotation_csv(args.primary, args.mapping)
            if args.secondary:
                annotations += read_annotation_csv(args.secondary, args.mapping, role="secondary")
            result = aggregate_human_scores(dataset, runs, annotations)
            root.parent.mkdir(parents=True, exist_ok=True)
            root.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result if args.command != "coverage" else {"output": str(root), "runs": len(runs)}, indent=2))
        return 0
    if args.command == "collect":
        from .collect import collect_sources
        query = Query(id=f"live-{args.company}-{args.as_of}", company=args.company, as_of=args.as_of,
                      window_days=args.days, focus=args.focus)
        sources = await collect_sources(query, audit_path=root / "collection-audit.json")
        query.source_ids = [s.id for s in sources][:8]
        output = root / "collected.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"schema_version": 1, "dataset_kind": "real",
            "sources": [s.model_dump(mode="json") for s in sources],
            "queries": [{**query.model_dump(mode="json"), "split": "dev"}]}, indent=2), encoding="utf-8")
        print(f"Saved {len(sources)} original-text sources: {output}")
        print("Review publication dates before research; unknown dates cannot establish current events.")
        return 0
    if args.command == "validate":
        from .evaluation import load_dataset, validate_dataset
        dataset = load_dataset(args.dataset)
        result = validate_dataset(dataset, strict_protocol=args.strict)
        print(json.dumps(result, indent=2, default=str))
        return int(bool(result))
    dataset, sources, queries = load_corpus(args.dataset)
    demo = args.command == "demo" or args.demo
    if demo and dataset.get("dataset_kind") != "synthetic_demo":
        raise ValueError("--demo requires an explicitly synthetic_demo dataset")
    if args.command != "batch":
        query = next((q for q in queries if q.id == args.query_id), None) if args.query_id else queries[0]
        if query is None:
            raise ValueError("Unknown query ID")
        record, _, output = await research(query, sources, variant=args.variant, demo=demo,
                                          dataset_kind=dataset.get("dataset_kind", "real"), output_dir=root)
        print(f"{record.status}: {output}")
        if record.error:
            print(record.error)
        return 1 if record.status == "failed" else 0
    from .retrieval import HashEncoder, MiniLMEncoder
    encoder = HashEncoder() if demo else MiniLMEncoder()
    rows = {q["id"]: q for q in dataset["queries"]}
    queries = [q for q in queries if args.split == "all" or rows[q.id].get("split") == args.split]
    if not queries:
        raise ValueError("No queries in selected split")
    runs, chunks_by_run = [], {}
    for query in queries:
        for variant in ("B0", "B1", "B2"):
            record, chunks, _ = await research(query, sources, variant=variant, demo=demo,
                dataset_kind=dataset.get("dataset_kind", "real"), output_dir=root, encoder=encoder)
            runs.append(record.model_dump(mode="json"))
            chunks_by_run[record.id] = [c.model_dump(mode="json") for c in chunks]
            print(f"{query.id} {variant}: {record.status}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch-runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (root / "batch-chunks.json").write_text(json.dumps(chunks_by_run, indent=2), encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in runs) for s in ("completed", "unverified", "failed")}
    summary = {"dataset_kind": dataset.get("dataset_kind"), "queries": len(queries), "runs": len(runs),
               "status_counts": counts, "human_quality_scores": None,
               "note": "Synthetic runs validate software only. No factual-quality claim without human annotation."}
    (root / "batch-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return int(counts["failed"] > 0)


def main():
    args = parser().parse_args()
    try:
        raise SystemExit(asyncio.run(execute(args)))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Cannot run: {exc}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
