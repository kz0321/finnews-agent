"""Application entry points shared by CLI and Streamlit."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .demo import DemoProvider
from .models import Query, Source
from .pipeline import Pipeline, render_brief
from .retrieval import HashEncoder, MiniLMEncoder, Retriever, build_chunks
from .store import EvidenceStore

ROOT = Path(__file__).resolve().parents[2]
DEMO_DATASET = ROOT / "data" / "demo" / "dataset.json"


def data_dir():
    load_dotenv(ROOT / ".env", override=False)
    path = Path(os.getenv("FINNEWS_DATA_DIR", str(ROOT / "runtime")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_corpus(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = [Source.model_validate({k: v for k, v in row.items() if k in Source.model_fields})
               for row in data["sources"]]
    queries = [Query.model_validate({k: v for k, v in row.items() if k in Query.model_fields})
               for row in data.get("queries", [])]
    return data, sources, queries


async def research(query, sources, *, variant="B2", demo=False, dataset_kind="real", output_dir=None,
                   encoder=None, provider=None):
    load_dotenv(ROOT / ".env", override=False)
    preparation_started = time.perf_counter()
    if demo and dataset_kind == "real":
        raise ValueError("Demo provider cannot produce research labeled real")
    root = Path(output_dir) if output_dir else data_dir()
    root.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(root / "evidence.sqlite")
    if not query.source_ids:
        raise ValueError("An explicit candidate pool is required")
    source_map = {s.id: s for s in sources}
    if len(source_map) != len(sources):
        raise ValueError("Source IDs must be unique in an imported corpus")
    if not set(query.source_ids).issubset(source_map):
        raise ValueError("Candidate pool contains unknown source IDs")
    pool = [source_map[i] for i in query.source_ids]
    if any(s.company != query.company for s in pool):
        raise ValueError("Candidate pool contains another company's source")
    encoder = encoder or (HashEncoder() if demo else MiniLMEncoder())
    chunks = []
    for source in pool:
        store.upsert_source(source)
        chunks.extend(build_chunks(source, tokenizer=encoder.tokenizer))
    store.save_chunks(chunks)
    retriever = Retriever(chunks, encoder, root / "vectors")
    if provider is None:
        if demo:
            provider = DemoProvider()
        else:
            from .providers import HostedProvider
            provider = HostedProvider.from_env()
    pipeline = Pipeline(retriever, provider, store=store)
    preparation_seconds = time.perf_counter() - preparation_started
    record = await pipeline.run(query, variant=variant, dataset_kind=dataset_kind)
    record.timings["preparation_seconds"] = round(preparation_seconds, 4)
    record.timings["retrieval"] = getattr(retriever, "events", [])
    store.save_run(record.model_dump(mode="json"))
    target = root / "runs" / record.id
    target.mkdir(parents=True, exist_ok=True)
    (target / "run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    (target / "chunks.json").write_text(json.dumps([c.model_dump(mode="json") for c in chunks], indent=2), encoding="utf-8")
    (target / "brief.md").write_text(render_brief(record, chunks), encoding="utf-8")
    return record, chunks, target
