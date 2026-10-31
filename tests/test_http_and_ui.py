"""Things in the web layer that can be tested without FastAPI: the security-headers middleware
(Starlette only) and the chat page's pure JavaScript helpers (via Node)."""
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "serving" / "static"


def call(app, path="/", method="GET"):
    """Minimal ASGI client: returns (status, headers, body)."""
    out = {"headers": {}, "body": b""}

    async def run():
        scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
                 "query_string": b"", "headers": [], "scheme": "http", "server": ("t", 80),
                 "client": ("1.1.1.1", 1), "http_version": "1.1", "root_path": ""}
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.sleep(3600)

        async def send(msg):
            if msg["type"] == "http.response.start":
                out["status"] = msg["status"]
                out["headers"] = {k.decode().lower(): v.decode() for k, v in msg["headers"]}
            elif msg["type"] == "http.response.body":
                out["body"] += msg.get("body", b"")

        await app(scope, receive, send)

    asyncio.run(run())
    return out["status"], out["headers"], out["body"]


@pytest.fixture
def secured():
    pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse, StreamingResponse
    from starlette.routing import Route

    async def stream(request):
        async def gen():
            for part in (b"event: token\n\n", b"event: done\n\n"):
                yield part
        return StreamingResponse(gen(), media_type="text/event-stream")

    from serving.http_security import SecurityHeaders
    app = Starlette(routes=[
        Route("/", lambda r: PlainTextResponse("hi")),
        Route("/stream", stream),
        Route("/docs", lambda r: PlainTextResponse("swagger")),
    ])
    return SecurityHeaders(app)


def test_every_response_gets_the_security_headers(secured):
    status, h, body = call(secured, "/")
    assert status == 200 and body == b"hi"
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "no-referrer"
    assert "default-src 'self'" in h["content-security-policy"] and "frame-ancestors 'none'" in h["content-security-policy"]


def test_streaming_responses_pass_through_intact_with_headers(secured):
    status, h, body = call(secured, "/stream")
    assert status == 200 and body == b"event: token\n\nevent: done\n\n"
    assert h["content-type"].startswith("text/event-stream") and h["x-content-type-options"] == "nosniff"


def test_swagger_docs_are_exempt_from_the_csp_only(secured):
    _, h, _ = call(secured, "/docs")
    assert "content-security-policy" not in h and h["x-content-type-options"] == "nosniff"


def test_the_page_has_no_inline_scripts_so_the_strict_csp_works():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    import re
    assert not re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html)
    assert not re.findall(r"\son\w+\s*=", html)                           # no onclick= etc.
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js


# ------------------------------------------------------------ JavaScript helpers via Node
def node(expr: str):
    out = subprocess.run(["node", "-e", f"const L=require('{(STATIC / 'lib.js').as_posix()}');"
                          f"console.log(JSON.stringify({expr}))"],
                         capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@needs_node
def test_js_sse_parser_handles_split_frames_crlf_and_junk():
    whole = node("""L.parseSSE('event: sources\\ndata: {"n":1}\\n\\nevent: token\\ndata: {"text":"a\\\\nb"}\\n\\n')""")
    assert [e["event"] for e in whole["events"]] == ["sources", "token"] and whole["rest"] == ""
    assert whole["events"][1]["data"]["text"] == "a\nb"

    partial = node("""L.parseSSE('event: token\\ndata: {"text":"x"}\\n\\nevent: tok')""")
    assert len(partial["events"]) == 1 and partial["rest"] == "event: tok"

    crlf = node("""L.parseSSE('event: done\\r\\ndata: {"ok":true}\\r\\n\\r\\n')""")
    assert crlf["events"][0]["data"] == {"ok": True}

    junk = node("""L.parseSSE('event: x\\ndata: {not json\\n\\nevent: y\\ndata: {"a":1}\\n\\n')""")
    assert [e["event"] for e in junk["events"]] == ["y"]                    # bad frame skipped


@needs_node
def test_js_citation_splitter_never_produces_markup():
    parts = node("""L.splitCitations('Do X [1]. Then <b>Y</b> [2, 3].')""")
    assert parts == [
        {"type": "text", "value": "Do X "}, {"type": "cite", "nums": [1]},
        {"type": "text", "value": ". Then <b>Y</b> "}, {"type": "cite", "nums": [2, 3]},
        {"type": "text", "value": "."}]                                       # '<b>' stays plain text
    assert node("L.splitCitations('no markers')") == [{"type": "text", "value": "no markers"}]
    assert node("L.splitCitations('[x] [] [1a]')") == [{"type": "text", "value": "[x] [] [1a]"}]


# ------------------------------------------------------------------ Step 14: request-size limit, HSTS, no-store

def raw_call(app, body_chunks=(b"",), headers=(), path="/", scheme="http"):
    """ASGI client that can send several body chunks; returns (status, headers, body)."""
    out = {"status": None, "headers": {}, "body": b""}

    async def run():
        scope = {"type": "http", "method": "POST", "path": path, "raw_path": path.encode(),
                 "query_string": b"", "headers": list(headers), "scheme": scheme,
                 "server": ("t", 80), "client": ("1.1.1.1", 1), "http_version": "1.1", "root_path": ""}
        chunks = list(body_chunks)

        async def receive():
            if chunks:
                chunk = chunks.pop(0)
                return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}
            await asyncio.sleep(3600)

        async def send(msg):
            if msg["type"] == "http.response.start":
                out["status"] = msg["status"]
                out["headers"] = {k.decode().lower(): v.decode() for k, v in msg["headers"]}
            elif msg["type"] == "http.response.body":
                out["body"] += msg.get("body", b"")

        await app(scope, receive, send)

    asyncio.run(run())
    return out["status"], out["headers"], out["body"]


def reading_app():
    """Reads the whole request body, then answers with how many bytes it saw."""
    async def app(scope, receive, send):
        size, more = 0, True
        while more:
            message = await receive()
            size += len(message.get("body", b""))
            more = message.get("more_body", False)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": str(size).encode()})
    return app


def test_small_bodies_pass_and_big_declared_bodies_are_refused_before_reading():
    from serving.http_security import BodyLimit

    app = BodyLimit(reading_app(), max_bytes=100)
    assert raw_call(app, [b"x" * 100], [(b"content-length", b"100")])[::2] == (200, b"100")
    status, headers, body = raw_call(app, [b"x" * 500], [(b"content-length", b"500")])
    assert status == 413 and headers["content-type"] == "application/json" and b"payload_too_large" in body


def test_chunked_bodies_without_content_length_are_counted():
    from serving.http_security import BodyLimit

    app = BodyLimit(reading_app(), max_bytes=100)
    assert raw_call(app, [b"x" * 60, b"x" * 30])[0] == 200
    assert raw_call(app, [b"x" * 60, b"x" * 60])[0] == 413          # 120 > 100 across two chunks


def test_a_nonsense_content_length_is_refused():
    from serving.http_security import BodyLimit

    assert raw_call(BodyLimit(reading_app(), 100), [b"x"], [(b"content-length", b"abc")])[0] == 413


def test_hsts_only_over_https_and_api_responses_are_not_cached():
    from serving.http_security import SecurityHeaders

    async def ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    app = SecurityHeaders(ok)
    _, plain, _ = raw_call(app, scheme="http")
    _, secure, _ = raw_call(app, scheme="https")
    assert "strict-transport-security" not in plain and "max-age=" in secure["strict-transport-security"]
    assert plain["cache-control"] == "no-store"
    assert "cache-control" not in raw_call(app, path="/static/app.js")[1]
