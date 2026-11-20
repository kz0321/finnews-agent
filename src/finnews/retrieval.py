"""Bounded, dated lexical/dense passage retrieval with original character offsets.

HashEncoder is an explicitly synthetic offline test substitute, not semantic retrieval.
MiniLMEncoder is the actual CPU semantic retriever used by non-demo runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .models import Chunk

TOKEN_PATTERN = re.compile(r"[$€£]?\d[\d,]*(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
CHUNK_VERSION = "sentence-offset-v1"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def financial_tokens(text):
    """Preserve financial amounts/percentages and currency symbols in lexical search."""
    return [m.group().lower().replace(",", "") if any(c.isdigit() for c in m.group())
            else m.group().lower() for m in TOKEN_PATTERN.finditer(text)]


class RegexTokenizer:
    """Network-free tokenizer for labelled demo fixtures only."""
    def encode(self, text, add_special_tokens=False):
        return financial_tokens(text)

    def __call__(self, text, **kwargs):
        matches = list(TOKEN_PATTERN.finditer(text))
        return {"input_ids": [m.group() for m in matches],
                "offset_mapping": [(m.start(), m.end()) for m in matches]}


class HashEncoder:
    model_id = "demo-hash-v1-384-NOT-SEMANTIC"
    tokenizer = RegexTokenizer()

    def encode(self, texts):
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in financial_tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % matrix.shape[1]
                matrix[row, index] += 1.0 if digest[4] & 1 else -1.0
        return _normalize(matrix)


class MiniLMEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", revision=None):
        from sentence_transformers import SentenceTransformer
        revision = revision or os.getenv("FINNEWS_EMBEDDING_REVISION") or MINILM_REVISION
        self.model = SentenceTransformer(model_name, device="cpu", revision=revision)
        self.model.max_seq_length = 256
        self.tokenizer = self.model.tokenizer
        config = self.model[0].auto_model.config
        resolved_revision = getattr(config, "_commit_hash", None) or revision or "local-unversioned"
        self.model_id = f"{model_name}@{resolved_revision}:max256:normalized:cpu"

    def encode(self, texts):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        for text in texts:
            if len(self.tokenizer.encode(text, add_special_tokens=True)) > 256:
                raise ValueError("Encoder input exceeds 256 tokens; silent truncation is forbidden")
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)


def _normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _token_length(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _short_title(title, tokenizer, budget=48):
    offsets = tokenizer(title, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= budget:
        return title
    return title[:offsets[budget - 1][1]]


def encoder_text(chunk, tokenizer):
    return _short_title(chunk.title, tokenizer) + "\n" + chunk.text


def build_chunks(source, tokenizer=None):
    """Split near sentence ends with <=180 tokens and up to 24 tokens of overlap.

    Offsets index the saved source text verbatim. No strip/normalization occurs here.
    The caller must use the same tokenizer as the intended encoder.
    """
    tokenizer = tokenizer or RegexTokenizer()
    offsets = tokenizer(source.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    offsets = [(int(a), int(b)) for a, b in offsets if b > a]
    if not offsets:
        return []
    boundaries = {m.end() for m in re.finditer(r"[.!?](?=\s|$)|\n", source.text)}
    result, first = [], 0
    while first < len(offsets):
        last = min(first + 180, len(offsets))
        if last < len(offsets):
            candidates = [i for i in range(first + 90, last + 1) if offsets[i - 1][1] in boundaries]
            if candidates:
                last = candidates[-1]
        start, end = offsets[first][0], offsets[last - 1][1]
        # Some tokenizers add context-sensitive tokens on re-encoding a substring.
        # Keep shrinking, rather than silently truncating or inventing offsets.
        title = _short_title(source.title, tokenizer)
        while last > first + 1 and (_token_length(tokenizer, source.text[start:end]) > 180 or
                len(tokenizer.encode(title + "\n" + source.text[start:end], add_special_tokens=True)) > 256):
            last -= 1
            end = offsets[last - 1][1]
        text = source.text[start:end]
        if _token_length(tokenizer, text) > 180:
            raise ValueError("A source token cannot fit in the chunk budget")
        identity = f"{CHUNK_VERSION}:{source.id}:{source.content_hash}:{start}:{end}"
        result.append(Chunk(id="c_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
            source_id=source.id, company=source.company, text=text, title=source.title,
            url=source.url, start=start, end=end, published_at=source.published_at,
            token_count=_token_length(tokenizer, text)))
        if last == len(offsets):
            break
        first = max(first + 1, last - 24)
    return result


def render_context(chunks):
    """Single shared evidence representation used for both budgeting and model input."""
    return json.dumps([{
        "id": c.id, "source_id": c.source_id, "title": c.title, "url": c.url,
        "published_at": str(c.published_at) if c.published_at else None,
        "background": c.background, "start": c.start, "end": c.end, "text": c.text,
    } for c in chunks], ensure_ascii=False, separators=(",", ":"))


_ENCODING = None


def context_token_count(context):
    """Count cl100k tokens when explicitly enabled, otherwise conservative UTF-8 bytes.

    Byte count is a safe upper bound for byte-pair tokenizers and needs no download.
    Set FINNEWS_TOKEN_COUNTER=cl100k to enable tiktoken (first use caches its asset).
    cl100k is a documented proxy, not the exact tokenizer of every hosted provider.
    """
    global _ENCODING
    if os.getenv("FINNEWS_TOKEN_COUNTER", "bytes") == "cl100k":
        import tiktoken
        if _ENCODING is None:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        return len(_ENCODING.encode(context, disallowed_special=()))
    return len(context.encode("utf-8"))


def _too_overlapping(chunk, selected):
    for other in selected:
        if chunk.source_id != other.source_id:
            continue
        overlap = max(0, min(chunk.end, other.end) - max(chunk.start, other.start))
        if overlap / max(1, min(chunk.end - chunk.start, other.end - other.start)) > 0.5:
            return True
    return False


def select_evidence(ranked_chunks, as_of, window_days, pinned_ids=(), limit=8, token_budget=2000):
    """Greedy shared caps with protected evidence; impossible pins fail explicitly."""
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    if window_days not in {7, 30} or not 0 <= limit <= 8 or token_budget < 0:
        raise ValueError("Invalid evidence selection limits")
    begin = as_of - timedelta(days=window_days - 1)
    pinned = set(pinned_ids)
    candidates = {}
    for chunk in ranked_chunks:
        if chunk.published_at is not None and chunk.published_at <= as_of:
            candidates.setdefault(chunk.id, chunk.model_copy(update={"background": chunk.published_at < begin}))
    if not pinned.issubset(candidates):
        raise ValueError("Pinned evidence is missing, undated or after the cutoff")
    ordered = [c for c in candidates.values() if c.id in pinned] + [c for c in candidates.values() if c.id not in pinned]
    selected, counts, backgrounds = [], Counter(), 0
    for chunk in ordered:
        allowed = (len(selected) < limit and counts[chunk.source_id] < 3
                   and (not chunk.background or backgrounds < 2)
                   and not _too_overlapping(chunk, selected)
                   and context_token_count(render_context(selected + [chunk])) <= token_budget)
        if not allowed:
            if chunk.id in pinned:
                raise ValueError("Pinned evidence cannot satisfy common count/diversity/token limits")
            continue
        selected.append(chunk)
        counts[chunk.source_id] += 1
        backgrounds += int(chunk.background)
    return selected


class Retriever:
    def __init__(self, chunks, encoder, cache_dir):
        self.chunks = list(chunks)
        self.encoder = encoder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._vectors = None
        self.cache_hit = False
        self.cached_vectors = 0
        self.encoded_vectors = 0
        self.events = []

    def _document_vectors(self):
        if self._vectors is not None:
            return self._vectors
        texts = [encoder_text(c, self.encoder.tokenizer) for c in self.chunks]
        vectors, missing, paths = [None] * len(texts), [], []
        for index, text in enumerate(texts):
            fingerprint = json.dumps({"encoder": self.encoder.model_id, "text": text}, ensure_ascii=False)
            cache = self.cache_dir / (hashlib.sha256(fingerprint.encode()).hexdigest() + ".npz")
            paths.append(cache)
            if cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as data:
                        vector = data["vector"]
                    if vector.ndim == 1 and vector.size == 384 and np.isfinite(vector).all():
                        vectors[index] = vector
                        self.cached_vectors += 1
                        continue
                except (OSError, ValueError, KeyError):
                    pass
            missing.append(index)
        if missing:
            generated = _normalize(self.encoder.encode([texts[i] for i in missing]))
            self.encoded_vectors += len(missing)
            for index, vector in zip(missing, generated, strict=True):
                vectors[index] = vector
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as handle:
                    temp_path = Path(handle.name)
                    np.savez_compressed(handle, vector=vector)
                try:
                    os.replace(temp_path, paths[index])
                finally:
                    temp_path.unlink(missing_ok=True)
        self.cache_hit = not missing
        self._vectors = _normalize(np.stack(vectors))
        return self._vectors

    def retrieve(self, query, as_of, window_days, mode="hybrid", pinned_ids=(), exclude_ids=(), limit=8):
        started = time.perf_counter()
        timing = {"mode": mode, "index_seconds": 0.0, "query_embedding_seconds": 0.0}
        if mode not in {"bm25", "hybrid"}:
            raise ValueError("Retrieval mode must be bm25 or hybrid")
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        excluded = set(exclude_ids)
        eligible = [i for i, c in enumerate(self.chunks) if c.id not in excluded
                    and c.published_at is not None and c.published_at <= as_of]
        if not eligible:
            if pinned_ids:
                raise ValueError("No eligible pinned evidence")
            timing.update(total_seconds=time.perf_counter() - started, eligible_chunks=0, selected_chunks=0)
            self.events.append(timing)
            return []
        lexical_started = time.perf_counter()
        documents = [financial_tokens(self.chunks[i].title + " " + self.chunks[i].text) or ["__empty__"] for i in eligible]
        lexical = BM25Okapi(documents).get_scores(financial_tokens(query))
        lexical_rank = sorted(range(len(eligible)), key=lambda j: (-float(lexical[j]), self.chunks[eligible[j]].id))[:20]
        timing["lexical_seconds"] = time.perf_counter() - lexical_started
        if mode == "bm25":
            ranks = lexical_rank
        else:
            embedding_started = time.perf_counter()
            query_vector = _normalize(self.encoder.encode([query]))[0]
            timing["query_embedding_seconds"] = time.perf_counter() - embedding_started
            index_started = time.perf_counter()
            vectors = self._document_vectors()
            timing["index_seconds"] = time.perf_counter() - index_started
            scores = vectors[eligible] @ query_vector
            dense_rank = sorted(range(len(eligible)), key=lambda j: (-float(scores[j]), self.chunks[eligible[j]].id))[:20]
            fused = {}
            for ranked in (lexical_rank, dense_rank):
                for position, j in enumerate(ranked, 1):
                    fused[j] = fused.get(j, 0) + 1 / (60 + position)
            # Equal RRF sums are common in tiny course candidate pools. Dense
            # similarity, then BM25 score, resolve ties before the stable ID.
            ranks = sorted(fused, key=lambda j: (-fused[j], -float(scores[j]),
                           -float(lexical[j]), self.chunks[eligible[j]].id))
        ranked = [self.chunks[eligible[j]] for j in ranks]
        # A pin may fall outside either top-20 list; keep it available explicitly.
        ranked.extend(c for c in self.chunks if c.id in set(pinned_ids) and c.id not in excluded)
        selection_started = time.perf_counter()
        selected = select_evidence(ranked, as_of, window_days, pinned_ids=pinned_ids, limit=limit)
        timing.update(selection_seconds=time.perf_counter() - selection_started,
                      total_seconds=time.perf_counter() - started, eligible_chunks=len(eligible),
                      selected_chunks=len(selected), cached_vectors=self.cached_vectors,
                      encoded_vectors=self.encoded_vectors)
        self.events.append(timing)
        return selected
