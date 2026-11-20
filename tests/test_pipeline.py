import json
from datetime import date

import pytest

from finnews.models import Brief, Chunk, Finding, Query
from finnews.pipeline import Pipeline, deterministic_checks, parse_json
from finnews.models import Verification


def chunk(cid="c1", source="s1", text="Revenue was 10 million US dollars."):
    return Chunk(id=cid, source_id=source, company="AAPL", title="Earnings", url="https://example.com/a",
                 text=text, start=0, end=len(text), token_count=12, published_at=date(2025, 1, 20))

QUERY = Query(id="q1", company="AAPL", as_of=date(2025, 1, 25), window_days=30, source_ids=["s1"])


def draft(text="Revenue was 10 million US dollars."):
    return Brief(company="AAPL", as_of=QUERY.as_of, window_days=30,
                 findings=[Finding(id="F1", text=text, evidence_ids=["c1"])])


def check(status="supported", quote="Revenue was 10 million US dollars.", queries=None):
    return {"findings": [{"finding_id": "F1", "status": status, "evidence_ids": ["c1"],
                           "quotation": quote, "reason": "test"}], "retrieval_queries": queries or []}


class ScriptedProvider:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.prompts = []

    async def generate(self, stage, prompt, evidence_context=""):
        self.calls.append({"stage": stage})
        self.prompts.append(prompt)
        value = next(self.outputs)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


class FakeRetriever:
    encoder = type("Encoder", (), {"model_id": "test-only"})()

    def retrieve(self, query, as_of, window_days, **kwargs):
        return [chunk("c2", "s2")] if kwargs.get("exclude_ids") else [chunk()]


@pytest.mark.asyncio
async def test_baseline_is_explicitly_unverified():
    provider = ScriptedProvider([draft().model_dump(mode="json")])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY, "B0", "synthetic_demo")
    assert record.status == "unverified"
    assert [c["stage"] for c in record.calls] == ["draft"]


@pytest.mark.asyncio
async def test_supported_run_stops_after_two_calls():
    provider = ScriptedProvider([draft().model_dump(mode="json"), check()])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY)
    assert record.status == "completed"
    assert len(record.calls) == 2
    assert record.brief.findings[0].id == "F1"


@pytest.mark.asyncio
async def test_wrong_number_cannot_pass_supported_label():
    provider = ScriptedProvider([draft("Revenue was 99 million US dollars.").model_dump(mode="json"),
                                 check(), {"patches": [], "remove_ids": ["F1"]}])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY)
    assert record.status == "completed"
    assert record.brief.findings == []
    assert "Information gap" in record.brief.limitations[0]


@pytest.mark.asyncio
async def test_recovery_and_revision_are_bounded():
    provider = ScriptedProvider([draft().model_dump(mode="json"),
        check("insufficient_evidence", "", ["revenue"]),
        {"patches": [draft().findings[0].model_dump()], "remove_ids": []}, check()])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY)
    assert record.status == "completed", record.error
    assert [c["stage"] for c in record.calls] == ["draft", "verify", "revise", "recheck"]
    assert len(set(record.selected_ids)-set(record.initial_selected_ids)) <= 2
    assert sum(t["state"] == "recover" for t in record.trace) == 1


@pytest.mark.asyncio
async def test_revision_cannot_invent_finding_ids():
    provider = ScriptedProvider([draft().model_dump(mode="json"), check("insufficient_evidence", ""),
        {"patches": [{"id": "F2", "text": "New claim", "evidence_ids": ["c1"]}], "remove_ids": []}])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY)
    assert record.status == "failed"
    assert "protected" in record.error


@pytest.mark.asyncio
async def test_only_one_schema_repair_for_whole_run():
    provider = ScriptedProvider(["bad", draft().model_dump(mode="json"), "bad"])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY)
    assert record.status == "failed"
    assert [c["stage"] for c in record.calls].count("repair") == 1


@pytest.mark.asyncio
async def test_timeout_retains_draft_as_unverified():
    provider = ScriptedProvider([draft().model_dump(mode="json"), TimeoutError()])
    record = await Pipeline(FakeRetriever(), provider).run(QUERY)
    assert record.status == "unverified"
    assert record.brief is not None


@pytest.mark.asyncio
async def test_unknown_citation_fails_closed():
    bad = draft().model_dump(mode="json")
    bad["findings"][0]["evidence_ids"] = ["not-selected"]
    record = await Pipeline(FakeRetriever(), ScriptedProvider([bad])).run(QUERY)
    assert record.status == "failed"


@pytest.mark.asyncio
async def test_uncited_background_cannot_bypass_verification():
    bad = draft().model_dump(mode="json")
    bad["background"] = ["An unverified financial claim."]
    record = await Pipeline(FakeRetriever(), ScriptedProvider([bad])).run(QUERY)
    assert record.status == "failed"
    assert "Background" in record.error


def test_json_parser_rejects_unstructured_explanations():
    assert parse_json('```json\n{"a":1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json('Explanation {"a":1}')


@pytest.mark.parametrize("claim,evidence", [
    ("Revenue was 10 million EUR.", "Revenue was 10 million USD."),
    ("Q2 revenue was 10 million USD.", "Q1 revenue was 10 million USD."),
    ("FY2025 revenue was 10 million USD.", "FY2024 revenue was 10 million USD."),
    ("Revenue growth was -10%.", "Revenue growth was 10%."),
])
def test_currencies_periods_and_signs_are_not_interchangeable(claim, evidence):
    verification = Verification.model_validate(check(quote=evidence))
    result = deterministic_checks(draft(claim), verification, [chunk(text=evidence)], QUERY)
    assert result[0].status == "insufficient_evidence"
