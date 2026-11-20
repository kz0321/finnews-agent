from datetime import date

import pytest

from finnews.models import Source
from finnews.retrieval import build_chunks
from finnews.store import EvidenceStore


def item():
    return Source(id="s", company="AAPL", title="Release", url="https://example.test/release",
                  text="Revenue increased 10 percent.", published_at=date(2025, 2, 10))


def test_roundtrip_original_spans_and_candidate_filter(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    src = item()
    chunks = build_chunks(src)
    store.upsert_source(src)
    store.save_chunks(chunks)
    assert store.get_sources(["s"])[0] == src
    assert store.get_chunks(["s"])[0] == chunks[0]
    assert store.get_chunks(["other"]) == []
    assert store.get_sources([]) == []
    assert store.get_sources(["' OR 1=1 --"]) == []
    bad = chunks[0].model_copy(update={"start": 1})
    with pytest.raises(ValueError, match="offsets"):
        store.save_chunks([bad])


def test_source_snapshot_cannot_be_overwritten(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    store.upsert_source(item())
    changed = item().model_dump()
    changed.update(text="New financial statement.", content_hash="")
    with pytest.raises(ValueError, match="different text snapshot"):
        store.upsert_source(Source.model_validate(changed))


def test_run_and_call_audit_is_atomic_and_idempotent(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    record = {"id": "run1", "variant": "B2", "status": "completed", "calls": [{"stage": "draft"}, {"stage": "verify"}]}
    store.save_run(record)
    store.save_run(record)
    assert store.get_run("run1") == record
    assert store.get_calls("run1") == record["calls"]
    assert store.get_run("missing") is None
    assert store.list_runs(limit=1) == [record]
    store.log_call({"run_id": "run2", "stage": "draft"})
    assert store.get_calls("run2")[0]["stage"] == "draft"
