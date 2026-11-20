from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from finnews.evaluation import (
    aggregate_human_scores,
    export_blind_annotations,
    load_dataset,
    reference_coverage,
    read_annotation_csv,
    retrieval_metrics,
    system_case,
    validate_dataset,
)


DATASET = Path(__file__).resolve().parents[1] / "data" / "demo" / "dataset.json"


def test_fixture_meets_shape_and_gold_never_enters_system_input():
    dataset = load_dataset(DATASET)
    assert validate_dataset(dataset, strict_protocol=True) == []
    query, sources = system_case(dataset, "aapl-test-08")
    assert "gold_facts" not in query.model_dump()
    assert "annotation_status" not in query.model_dump()
    assert len(sources) == 4
    assert all(source.content_hash for source in sources)


def test_reference_span_can_cross_chunks_but_gaps_do_not_count():
    gold = [{"id": "fact", "evidence": [{"source_id": "s", "start": 10, "end": 30}]}]
    chunks = [{"source_id": "s", "start": 0, "end": 20}, {"source_id": "s", "start": 20, "end": 40}]
    assert reference_coverage(gold, chunks)["ratio"] == 1
    chunks[1]["start"] = 21
    assert reference_coverage(gold, chunks)["ratio"] == 0
    chunks[1]["source_id"] = "different"
    assert reference_coverage(gold, chunks)["ratio"] == 0


def test_invalid_gold_offsets_and_future_gold_rejected():
    dataset = load_dataset(DATASET)
    query = dataset["queries"][0]
    query["gold_facts"][0]["evidence"][0]["end"] = 1_000_000
    assert any("invalid original-text offsets" in error for error in validate_dataset(dataset))
    source = next(s for s in dataset["sources"] if s["id"] == query["source_ids"][0])
    source["published_at"] = "2099-01-01"
    assert any("pre-cutoff" in error for error in validate_dataset(dataset))


def _run(dataset, variant="B2", status="completed"):
    query = next(q for q in dataset["queries"] if q["split"] == "test")
    return {
        "id": "run-1", "query": {k: query[k] for k in ("id", "company", "as_of", "window_days", "focus", "source_ids")},
        "variant": variant, "status": status, "dataset_kind": "synthetic_demo",
        "brief": {"findings": [{"id": "F1", "text": "A fictional fact.", "evidence_ids": ["C1"]}]},
        "checks": [{"finding_id": "F1", "status": "supported"}], "calls": [], "timings": {},
    }


def test_unscored_verifier_supported_is_not_human_truth():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    scores = aggregate_human_scores(dataset, [run], [])
    result = scores["variants"]["B2"]
    assert result["supported_and_cited_claim_ratio"] is None
    assert result["key_fact_coverage_all_queries"] is None
    assert result["unscored_reports"] == 1
    assert result["missing_runs"] == 19


def test_failed_and_missing_queries_stay_in_denominator():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    gold_id = next(q for q in dataset["queries"] if q["id"] == run["query"]["id"])["gold_facts"][0]["id"]
    annotation = {"run_id": run["id"], "finding_id": "F1", "claim_id": "F1.1", "supported": 1, "temporal_correct": 1, "covered_gold_fact_ids": [gold_id], "annotator": "A", "role": "primary"}
    result = aggregate_human_scores(dataset, [run], [annotation])["variants"]["B2"]
    assert result["key_fact_coverage_all_queries"] == pytest.approx(1 / 3 / 20)
    assert result["supported_and_cited_claim_ratio"] == 1
    failed = deepcopy(run)
    failed["status"] = "failed"
    result = aggregate_human_scores(dataset, [failed], [annotation])["variants"]["B2"]
    assert result["key_fact_coverage_all_queries"] == 0
    assert result["supported_and_cited_claim_ratio"] is None


def test_unknown_timing_gets_zero_temporal_credit():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    annotation = {"run_id": run["id"], "finding_id": "F1", "claim_id": "F1.1", "supported": 1, "temporal_correct": "unknown", "covered_gold_fact_ids": [], "annotator": "A"}
    result = aggregate_human_scores(dataset, [run], [annotation])["variants"]["B2"]
    assert result["temporal_correct_ratio"] == 0
    assert result["unknown_timing_claims"] == 1


def test_export_is_blinded_and_assigns_at_least_twenty_percent(tmp_path):
    dataset = load_dataset(DATASET)
    runs = []
    for i in range(6):
        run = _run(dataset)
        run["id"] = f"run-{i}"
        runs.append(run)
    result = export_blind_annotations(runs, tmp_path)
    assert result["double_scored_reports_requested"] == 2
    primary = (tmp_path / "primary_claims.csv").read_text(encoding="utf-8-sig")
    assert "B2" not in primary
    assert "run-" not in primary
    assert "supported" in primary
    with pytest.raises(ValueError):
        export_blind_annotations(runs, tmp_path, double_score_fraction=0.1)


def test_double_scoring_requires_distinct_people():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    primary = {"run_id": run["id"], "finding_id": "F1", "claim_id": "F1.1", "supported": 1, "temporal_correct": 1, "covered_gold_fact_ids": [], "annotator": "A", "role": "primary"}
    secondary = {**primary, "role": "secondary"}
    with pytest.raises(ValueError, match="two different"):
        aggregate_human_scores(dataset, [run], [primary, secondary])


def test_agreement_uses_original_independent_labels_before_adjudication():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    primary = {"run_id": run["id"], "finding_id": "F1", "claim_id": "F1.1", "supported": 1, "temporal_correct": 1, "covered_gold_fact_ids": [], "annotator": "A", "role": "primary"}
    secondary = {**primary, "annotator": "B", "role": "secondary", "supported": 0}
    adjudicated = {**primary, "annotator": "C", "role": "adjudicated", "supported": 0}
    result = aggregate_human_scores(dataset, [run], [primary, secondary, adjudicated])
    assert result["agreement"]["raw_support_agreement"] == 0
    assert result["variants"]["B2"]["supported_and_cited_claim_ratio"] == 0


def test_retrieval_metric_uses_selected_spans_and_full_query_denominator():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    query = next(q for q in dataset["queries"] if q["id"] == run["query"]["id"])
    source = next(s for s in dataset["sources"] if s["id"] == query["source_ids"][0])
    run["initial_selected_ids"] = []
    run["selected_ids"] = ["C1"]
    chunks = {run["id"]: [{"id": "C1", "source_id": source["id"], "start": 0, "end": len(source["text"])}]}
    result = retrieval_metrics(dataset, [run], chunks)["variants"]["B2"]
    assert result["initial_coverage"] == 0
    assert result["final_coverage"] == pytest.approx(1 / 20)
    query["gold_facts"] = []
    assert retrieval_metrics(dataset, [run], chunks)["variants"]["B2"]["final_coverage"] is None


def test_unscored_second_atomic_claim_cannot_disappear_from_csv(tmp_path):
    path = tmp_path / "claims.csv"
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps([{"blind_id": "R001", "run_id": "run-1"}]))
    fields = ["blind_id", "finding_id", "claim_id", "supported", "temporal_correct", "covered_gold_fact_ids", "annotator"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"blind_id": "R001", "finding_id": "F1", "claim_id": "F1.1", "supported": "1", "temporal_correct": "1", "annotator": "A"})
        writer.writerow({"blind_id": "R001", "finding_id": "F1", "claim_id": "F1.2"})
    with pytest.raises(ValueError, match="every atomic claim"):
        read_annotation_csv(path, mapping)


def test_partial_double_score_is_not_a_double_scored_report():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    p1 = {"run_id": run["id"], "finding_id": "F1", "claim_id": "F1.1", "supported": 1, "temporal_correct": 1, "covered_gold_fact_ids": [], "annotator": "A", "role": "primary"}
    p2 = {**p1, "claim_id": "F1.2"}
    secondary = {**p1, "annotator": "B", "role": "secondary"}
    result = aggregate_human_scores(dataset, [run], [p1, p2, secondary])
    assert result["agreement"]["paired_atomic_claims"] == 1
    assert result["agreement"]["double_scored_reports"] == 0


def test_unknown_run_labels_rejected_and_dev_labels_excluded():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    dev = deepcopy(run)
    dev["id"] = "dev-run"
    dev["query"]["id"] = next(q for q in dataset["queries"] if q["split"] == "dev")["id"]
    label = {"run_id": "dev-run", "finding_id": "F1", "claim_id": "F1.1", "supported": 1, "temporal_correct": 1, "covered_gold_fact_ids": [], "annotator": "A", "role": "primary"}
    second = {**label, "annotator": "B", "role": "secondary"}
    result = aggregate_human_scores(dataset, [run, dev], [label, second])
    assert result["agreement"]["paired_atomic_claims"] == 0
    assert result["agreement"]["double_scored_reports"] == 0
    with pytest.raises(ValueError, match="unknown run ID"):
        aggregate_human_scores(dataset, [run], [label])


def test_cost_scope_is_llm_only_and_missing_runs_have_unknown_total():
    dataset = load_dataset(DATASET)
    run = _run(dataset)
    run["calls"] = [{"cost_usd": 0.25}]
    result = aggregate_human_scores(dataset, [run], [])["variants"]["B2"]
    assert result["known_llm_call_cost_usd"] == 0.25
    assert result["llm_call_cost_usd"] is None
    assert result["total_cost_usd"] is None
    empty = aggregate_human_scores(dataset, [], [])["variants"]["B2"]
    assert empty["total_cost_usd"] is None


def test_overlapping_dev_test_windows_rejected_even_with_different_cutoffs():
    dataset = load_dataset(DATASET)
    dev = next(q for q in dataset["queries"] if q["company"] == "AAPL" and q["split"] == "dev")
    test = next(q for q in dataset["queries"] if q["company"] == "AAPL" and q["split"] == "test")
    test["as_of"] = "2025-07-15"
    test["window_days"] = 7
    assert dev["as_of"] != test["as_of"]
    assert any("Overlapping development/test" in error for error in validate_dataset(dataset))


def test_official_aspnet_form_wrapper_does_not_delete_disclosure():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_real_corpus.py"
    spec = importlib.util.spec_from_file_location("finnews_test_corpus_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = "Original fictional revenue disclosure with its reporting period and currency. " * 12
    html = f"<html><body><form><nav>Navigation</nav><article><h1>Results</h1><p>{body}</p><table><tr><td>Revenue</td><td>2024</td><td>100</td></tr></table></article></form></body></html>"
    text = module.extract_text(html)
    assert body.strip() in text
    assert "Revenue | 2024 | 100" in text
    assert "Navigation" not in text
