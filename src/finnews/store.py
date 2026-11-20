"""SQLite evidence and execution records; original text is never reconstructed from chunks."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import Chunk, Source


def _json(record):
    if hasattr(record, "model_dump"):
        record = record.model_dump(mode="json")
    return json.dumps(record, ensure_ascii=False, default=str)


class EvidenceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, company TEXT NOT NULL, url TEXT NOT NULL,
                    content_hash TEXT NOT NULL, record_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS source_hash ON sources(content_hash);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                    start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
                    record_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunk_source ON chunks(source_id);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    variant TEXT, status TEXT, record_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    record_json TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def upsert_source(self, source: Source):
        # IDs identify immutable text snapshots. Reusing one for new text would
        # invalidate saved offsets and citations in historical runs.
        with self._connect() as db:
            old = db.execute("SELECT content_hash FROM sources WHERE id=?", (source.id,)).fetchone()
            if old and old[0] != source.content_hash:
                raise ValueError("Source ID already belongs to a different text snapshot")
            db.execute("""INSERT INTO sources VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET company=excluded.company,url=excluded.url,
                record_json=excluded.record_json""",
                (source.id, source.company, source.url, source.content_hash, _json(source)))

    def save_chunks(self, chunks):
        with self._connect() as db:
            source_text = {}
            for chunk in chunks:
                if chunk.source_id not in source_text:
                    row = db.execute("SELECT record_json FROM sources WHERE id=?", (chunk.source_id,)).fetchone()
                    if not row:
                        raise ValueError("Save a source before its chunks")
                    source_text[chunk.source_id] = json.loads(row[0])["text"]
                if source_text[chunk.source_id][chunk.start:chunk.end] != chunk.text:
                    raise ValueError("Chunk offsets do not reproduce the original saved source span")
                db.execute("""INSERT INTO chunks VALUES(?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET record_json=excluded.record_json""",
                    (chunk.id, chunk.source_id, chunk.start, chunk.end, _json(chunk)))

    def get_sources(self, ids=None):
        return self._records("sources", Source, "id", ids)

    def get_chunks(self, source_ids=None):
        return self._records("chunks", Chunk, "source_id", source_ids)

    def _records(self, table, model, key, values):
        if values is not None:
            values = list(values)
            if not values:
                return []
        sql = f"SELECT record_json FROM {table}"
        if values is not None:
            sql += f" WHERE {key} IN ({','.join('?' for _ in values)})"
        sql += " ORDER BY rowid"
        with self._connect() as db:
            return [model.model_validate_json(row[0]) for row in db.execute(sql, values or [])]

    def save_run(self, record):
        data = json.loads(_json(record))
        with self._connect() as db:
            db.execute("""INSERT INTO runs(id,variant,status,record_json) VALUES(?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET variant=excluded.variant,status=excluded.status,
                record_json=excluded.record_json""",
                (data["id"], data.get("variant"), data.get("status"), _json(data)))
            # Replace a run's audit rows atomically, so re-saving cannot double-count.
            db.execute("DELETE FROM calls WHERE run_id=?", (data["id"],))
            for call in data.get("calls", []):
                db.execute("INSERT INTO calls(run_id,record_json) VALUES(?,?)", (data["id"], _json(call)))

    def log_call(self, record):
        data = json.loads(_json(record))
        with self._connect() as db:
            db.execute("INSERT INTO calls(run_id,record_json) VALUES(?,?)", (data.get("run_id"), _json(data)))

    def get_run(self, run_id):
        with self._connect() as db:
            row = db.execute("SELECT record_json FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(self, limit=100):
        with self._connect() as db:
            rows = db.execute("SELECT record_json FROM runs ORDER BY rowid DESC LIMIT ?", (max(0, int(limit)),))
            return [json.loads(row[0]) for row in rows]

    def get_calls(self, run_id):
        with self._connect() as db:
            rows = db.execute("SELECT record_json FROM calls WHERE run_id=? ORDER BY id", (run_id,))
            return [json.loads(row[0]) for row in rows]
