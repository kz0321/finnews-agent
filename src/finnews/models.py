"""Shared, strict wire schemas. Gold annotations never enter these models."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOGL": "Alphabet"}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Record):
    id: str
    company: str
    url: str
    title: str
    text: str
    published_at: date | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""

    @model_validator(mode="after")
    def hash_text(self):
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash and self.content_hash != digest:
            raise ValueError("content_hash does not match saved source text")
        self.content_hash = digest
        return self


class Chunk(Record):
    id: str
    source_id: str
    company: str
    text: str
    title: str
    url: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    published_at: date | None = None
    token_count: int = Field(ge=0)
    background: bool = False


class Query(Record):
    id: str
    company: Literal["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    as_of: date
    window_days: Literal[7, 30] = 30
    focus: str = "earnings, material announcements and risks"
    source_ids: list[str] = Field(default_factory=list, max_length=8)


class Finding(Record):
    id: str
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class Brief(Record):
    company: str
    as_of: date
    window_days: Literal[7, 30]
    findings: list[Finding] = Field(default_factory=list, max_length=5)
    background: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_ids(self):
        ids = [f.id for f in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding IDs must be unique")
        return self


class FindingCheck(Record):
    finding_id: str
    status: Literal["supported", "contradicted", "insufficient_evidence"]
    evidence_ids: list[str] = Field(default_factory=list)
    quotation: str = ""
    reason: str = ""


class Verification(Record):
    findings: list[FindingCheck]
    retrieval_queries: list[str] = Field(default_factory=list, max_length=2)


class Revision(Record):
    patches: list[Finding] = Field(default_factory=list, max_length=5)
    remove_ids: list[str] = Field(default_factory=list, max_length=5)


class RunRecord(Record):
    id: str
    query: Query
    variant: Literal["B0", "B1", "B2"]
    status: Literal["completed", "unverified", "failed"] = "failed"
    dataset_kind: str
    encoder: str
    initial_selected_ids: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    brief: Brief | None = None
    checks: list[FindingCheck] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    calls: list[dict] = Field(default_factory=list)
    timings: dict = Field(default_factory=dict)
    error: str | None = None
