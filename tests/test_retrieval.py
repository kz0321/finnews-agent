from datetime import date, timedelta

import numpy as np
import pytest

from finnews.models import Source
from finnews.retrieval import (HashEncoder, RegexTokenizer, Retriever, build_chunks,
                              context_token_count, encoder_text, financial_tokens,
                              render_context, select_evidence)


def source(sid="s1", text="Revenue increased to $10.5 billion in Q1 2025.", day=date(2025, 2, 10)):
    return Source(id=sid, company="AAPL", url=f"https://example.test/{sid}", title="Earnings",
                  text=text, published_at=day)


def test_chunks_preserve_offsets_caps_and_stable_ids():
    item = source(text="  " + "Revenue grew by 20 percent. " * 300)
    chunks = build_chunks(item)
    assert len(chunks) > 2
    assert [c.id for c in chunks] == [c.id for c in build_chunks(item)]
    assert all(item.text[c.start:c.end] == c.text for c in chunks)
    assert all(c.token_count <= 180 for c in chunks)
    assert all(c.start < c.end for c in chunks)
    assert chunks[0].start == 2
    assert all(right.start < left.end for left, right in zip(chunks, chunks[1:]))
    assert chunks[-1].end == len(item.text.rstrip())


def test_long_title_is_bounded_for_encoder_but_retained_as_metadata():
    item = source(text="Profit rose. " * 160)
    item.title = "A long title " * 100
    chunks = build_chunks(item)
    assert all(c.title == item.title for c in chunks)
    assert all(len(RegexTokenizer().encode(encoder_text(c, RegexTokenizer()))) <= 230 for c in chunks)


def test_financial_tokenization_retains_amounts_and_percentages():
    assert financial_tokens("EPS $1,234.50, growth 10.2% and Q1 FY2025") == ["eps", "$1234.50", ",", "growth", "10.2%", "and", "q1", "fy2025"]


def test_cutoff_window_background_source_diversity_and_token_budget(monkeypatch):
    monkeypatch.setenv("FINNEWS_TOKEN_COUNTER", "bytes")
    today = date(2025, 2, 10)
    rows = []
    for i in range(11):
        day = today if i < 7 else today - timedelta(days=20)
        rows += build_chunks(source(f"s{i}", "A small financial fact.", day))
    rows += build_chunks(source("future", day=today + timedelta(days=1)))
    rows += build_chunks(source("unknown", day=None))
    selected = select_evidence(rows, today, 7, token_budget=10000)
    assert len(selected) == 8
    assert sum(c.background for c in selected) <= 2
    assert all(c.published_at <= today for c in selected)
    assert not {"future", "unknown"} & {c.source_id for c in selected}
    actual = select_evidence(rows, today, 7)
    assert context_token_count(render_context(actual)) <= 2000
    assert actual[0].background is False
    assert rows[-3].background is False  # selection never mutates stored chunks


def test_pinned_evidence_is_never_silently_dropped():
    chunks = build_chunks(source())
    with pytest.raises(ValueError, match="Pinned evidence"):
        select_evidence(chunks, date(2025, 2, 10), 7, pinned_ids=[chunks[0].id], token_budget=10)
    with pytest.raises(ValueError, match="Pinned evidence"):
        select_evidence(chunks, date(2025, 2, 10), 7, pinned_ids=["missing"])


def test_overlap_and_per_source_caps():
    chunks = build_chunks(source(text="Revenue grew. " * 500))
    selected = select_evidence(chunks, date(2025, 2, 10), 7, token_budget=100000)
    assert len(selected) == 3
    duplicate = chunks[0].model_copy(update={"id": "different-id"})
    assert len(select_evidence([chunks[0], duplicate], date(2025, 2, 10), 7, token_budget=10000)) == 1


def test_retrieval_respects_pool_exclusions_and_cache(tmp_path):
    chunks = build_chunks(source("growth", "Revenue increased to $10.5 billion."))
    chunks += build_chunks(source("risk", "A hurricane disrupted manufacturing operations."))
    chunks += build_chunks(source("future", day=date(2025, 3, 1)))
    encoder = HashEncoder()
    initial = Retriever(chunks, encoder, tmp_path)
    result = initial.retrieve("hurricane manufacturing", date(2025, 2, 10), 7, limit=1)
    assert result[0].source_id == "risk"
    assert initial.cache_hit is False
    replay = Retriever(chunks, encoder, tmp_path)
    result = replay.retrieve("hurricane manufacturing", date(2025, 2, 10), 7,
                             exclude_ids=[result[0].id])
    assert replay.cache_hit is True
    assert {c.source_id for c in result} == {"growth"}
    assert len(list(tmp_path.glob("*.npz"))) == 3
    # The same source passage is reusable when another query has a different pool.
    subset = Retriever(chunks[:1], encoder, tmp_path)
    subset.retrieve("revenue", date(2025, 2, 10), 7)
    assert subset.cache_hit is True
    assert subset.encoded_vectors == 0
    mutated = chunks[0].model_copy(update={"text": "Changed snapshot."})
    Retriever([mutated], encoder, tmp_path).retrieve("snapshot", date(2025, 2, 10), 7)
    assert len(list(tmp_path.glob("*.npz"))) == 4


def test_demo_encoder_is_deterministic_and_labelled():
    encoder = HashEncoder()
    assert "NOT-SEMANTIC" in encoder.model_id
    assert np.array_equal(encoder.encode(["revenue"]), encoder.encode(["revenue"]))
    assert np.allclose(np.linalg.norm(encoder.encode(["revenue", "risk"]), axis=1), 1)


def test_empty_pool_returns_no_evidence(tmp_path):
    assert Retriever([], HashEncoder(), tmp_path).retrieve("any", date(2025, 2, 10), 7) == []
