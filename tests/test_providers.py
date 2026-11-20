import asyncio
import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from finnews.collect import allowed_url, collect_sources, metadata_date
from finnews.models import Query
from finnews.providers import HostedProvider, ProviderConfig, ProviderError


def test_deepseek_env_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "private-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    config = ProviderConfig.from_env()
    assert config.model == "deepseek-flash"
    assert config.base_url == "https://api.deepseek.com"
    assert "private-test-key" not in repr(config)


async def test_http_usage_and_one_retry():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, json={"error": "private-server-body"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    provider = HostedProvider(ProviderConfig("test", input_price=1, output_price=2),
                              transport=httpx.MockTransport(handler))
    assert await provider.generate("verify", "JSON please") == '{"ok":true}'
    assert len(provider.calls) == 2
    assert provider.calls[0]["error"] == "HTTP_429"
    assert provider.calls[1]["cost_usd"] == pytest.approx(0.00014)
    assert requests[1]["model"] == "deepseek-flash"
    assert requests[1]["response_format"] == {"type": "json_object"}
    assert requests[1]["thinking"] == {"type": "disabled"}


async def test_no_retry_on_auth_error_and_no_body_leak():
    provider = HostedProvider(ProviderConfig("secret-key"), transport=httpx.MockTransport(
        lambda req: httpx.Response(401, text="Your secret-key is invalid")))
    with pytest.raises(ProviderError) as err:
        await provider.generate("verify", "check")
    assert "secret-key" not in str(err.value)
    assert len(provider.calls) == 1
    assert provider.calls[0]["cost_usd"] is None


async def test_maximum_two_network_attempts():
    provider = HostedProvider(ProviderConfig("secret"), transport=httpx.MockTransport(
        lambda req: httpx.Response(503)))
    with pytest.raises(ProviderError):
        await provider.generate("verify", "check")
    assert len(provider.calls) == 2


async def test_protocol_error_and_timeout_are_recorded():
    def broken(request):
        raise httpx.RemoteProtocolError("server disconnected")

    provider = HostedProvider(ProviderConfig("secret"), transport=httpx.MockTransport(broken))
    with pytest.raises(ProviderError):
        await provider.generate("verify", "check")
    assert [c["error"] for c in provider.calls] == ["RemoteProtocolError", "RemoteProtocolError"]

    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200)

    provider = HostedProvider(ProviderConfig("secret"), transport=httpx.MockTransport(slow))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.generate("verify", "check"), timeout=0.01)
    assert provider.calls[0]["error"] == "CancelledOrTimedOut"
    assert provider.calls[0]["latency_seconds"] is not None


def test_publication_metadata_and_host_boundaries():
    assert metadata_date('<meta property="article:published_time" content="2025-01-30T10:00:00Z">') == date(2025, 1, 30)
    assert metadata_date('<script type="application/ld+json">{"dateModified":"2025-01-30"}</script>') is None
    assert metadata_date('<script type="application/ld+json">{"datePublished":"2025-01-30"}</script>') == date(2025, 1, 30)
    assert metadata_date('<meta name="date" content="2025-01-30"><meta name="date" content="2025-02-01">') is None
    assert allowed_url("https://investor.apple.com/release", ["apple.com"])
    for bad in ("https://apple.com.evil.test/a", "https://apple.com@evil.test/a", "http://apple.com/a",
                "https://apple.com:8080/a", "https://apple.com.invalid/a"):
        assert not allowed_url(bad, ["apple.com"])


async def test_collection_never_uses_search_snippets(monkeypatch, tmp_path):
    pytest.importorskip("langchain_core")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    text = "Apple announced revenue of $100 billion for its fiscal quarter. " * 4
    rows = [
        {"url": "https://apple.com/snippet", "content": text},
        {"url": "https://apple.com/release", "raw_content": text, "title": "Release"},
        {"url": "https://apple.com/copy", "raw_content": text},
        {"url": "https://apple.com.evil.test/release", "raw_content": text + "invalid"},
        {"url": "https://apple.com/future", "raw_content": text + "future"},
    ]

    class Researcher:
        def __init__(self, **kwargs):
            self.cfg = SimpleNamespace()
            self.kwargs = kwargs

        async def conduct_research(self):
            assert self.cfg.context_filter == "none"
            assert self.cfg.fast_llm_model == self.cfg.smart_llm_model == self.cfg.strategic_llm_model

        def get_research_sources(self):
            return rows

    def response(request):
        value = "2025-02-20" if request.url.path == "/future" else "2025-01-30"
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text=f'<meta property="article:published_time" content="{value}">')

    path = tmp_path / "audit.json"
    query = Query(id="q", company="AAPL", as_of=date(2025, 1, 31))
    sources = await collect_sources(query, config=ProviderConfig("secret"), researcher_factory=Researcher,
                                    transport=httpx.MockTransport(response), audit_path=path)
    assert len(sources) == 1
    assert sources[0].published_at == date(2025, 1, 30)
    assert sources[0].text == text.strip()
    assert "secret" not in path.read_text()


@pytest.fixture
def compatible_server():
    """Real upstream HTTP/stream integration; never contacts a paid service."""
    requests = []
    status = {"value": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if status["value"] != 200:
                self.send_response(status["value"])
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"unavailable","type":"server_error"}}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "test", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-flash",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"ok":true}'},
                              "finish_reason": None}]},
                {"id": "test", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-flash",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}},
            ]
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests, status
    server.shutdown()
    server.server_close()
    thread.join()


async def test_real_gptr_write_report_deepseek_compatible(compatible_server, monkeypatch):
    pytest.importorskip("gpt_researcher")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    base_url, requests, _ = compatible_server
    provider = HostedProvider(ProviderConfig("not-a-real-key", base_url=base_url))
    result = await provider.generate("draft", "Return JSON only: {ok: true}", "[evidence-1] original text")
    assert json.loads(result) == {"ok": True}
    assert len(requests) == len(provider.calls) == 1
    assert requests[0]["model"] == "deepseek-flash"
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "[evidence-1] original text" in requests[0]["messages"][-1]["content"]
    assert provider.calls[0]["input_tokens"] == 100
    assert provider.calls[0]["output_tokens"] == 5
    assert provider.calls[0]["cost_usd"] is None


async def test_real_gptr_draft_has_at_most_one_retry(compatible_server, monkeypatch):
    pytest.importorskip("gpt_researcher")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    base_url, requests, status = compatible_server
    status["value"] = 503
    provider = HostedProvider(ProviderConfig("not-a-real-key", base_url=base_url))
    with pytest.raises(ProviderError):
        await provider.generate("draft", "Return JSON", "evidence")
    assert len(requests) == len(provider.calls) == 2
    assert all(call["error"] for call in provider.calls)
