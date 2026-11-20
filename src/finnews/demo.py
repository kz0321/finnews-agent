"""Explicitly synthetic, deterministic provider. Never presented as an LLM."""
from __future__ import annotations

import json
import re
import time


class DemoProvider:
    def __init__(self):
        self.calls = []

    async def generate(self, stage, prompt, evidence_context=""):
        started = time.perf_counter()
        payload = json.loads(prompt.split("\nPAYLOAD\n", 1)[1])
        context = json.loads(evidence_context)
        if isinstance(context, dict):
            context = context.get("evidence", context.get("chunks", []))
        chunks = {c.get("id", c.get("chunk_id")): c for c in context}
        if stage == "draft":
            query = payload["query"]
            findings = []
            for cid, chunk in chunks.items():
                if chunk.get("background"):
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
                for sentence in sentences:
                    if sentence.startswith("Synthetic classroom") or sentence.startswith("This background"):
                        continue
                    if len(sentence.split()) < 7:
                        continue
                    findings.append({"id": f"F{len(findings)+1}", "text": sentence, "evidence_ids": [cid]})
                    if len(findings) == 5:
                        break
                if len(findings) == 5:
                    break
            value = {"company": query["company"], "as_of": query["as_of"], "window_days": query["window_days"],
                     "findings": findings, "background": [],
                     "limitations": ["Synthetic fixture with deterministic extraction; no LLM or real market evaluation."]}
        elif stage in {"verify", "recheck"}:
            checks = []
            for finding in payload["brief"]["findings"]:
                matches = [cid for cid in finding["evidence_ids"] if cid in chunks and finding["text"] in chunks[cid]["text"]]
                checks.append({"finding_id": finding["id"], "status": "supported" if matches else "insufficient_evidence",
                    "evidence_ids": matches, "quotation": finding["text"] if matches else "",
                    "reason": "Demo exact-span check only; not a semantic model judgment."})
            value = {"findings": checks, "retrieval_queries": []}
        elif stage == "revise":
            value = {"patches": [], "remove_ids": payload["flagged_ids"]}
        else:
            raise ValueError("Demo provider has no schema-repair model")
        self.calls.append({"stage": stage, "attempt": 1, "provider": "deterministic_demo", "model": "none",
                           "latency_seconds": time.perf_counter()-started, "input_tokens": 0,
                           "output_tokens": 0, "cost_usd": 0.0, "actual_network_request": False})
        return json.dumps(value)
