"""Security headers for every response, as a pure ASGI middleware (it never buffers, so it is
safe for streaming responses, unlike Starlette's BaseHTTPMiddleware).

The chat page is served from this same app and uses no inline scripts, so the CSP can forbid
everything except our own files. Swagger UI (/docs) loads scripts from a CDN, so it is exempt.
"""
from starlette.datastructures import MutableHeaders

CSP = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class SecurityHeaders:
    def __init__(self, app, frame_origin: str = ""):
        self.app = app
        # Only the admin page may embed anything, and only the configured Grafana origin.
        self.admin_csp = f"{CSP}; frame-src {frame_origin}" if frame_origin else CSP

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("Referrer-Policy", "no-referrer")
                if scope.get("scheme") == "https":      # only meaningful (and only sent) over TLS
                    headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
                if not path.startswith("/static"):       # API answers can contain internal text
                    headers.setdefault("Cache-Control", "no-store")
                if not path.startswith(DOCS_PATHS):
                    headers.setdefault("Content-Security-Policy",
                                       self.admin_csp if path == "/admin" else CSP)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _TooLarge(Exception):
    pass


class BodyLimit:
    """Reject request bodies over `max_bytes` with 413 before the framework buffers them.

    Questions are capped at 500 characters and logins are tiny, so 16 KB is generous. Without a
    limit, one client could make the server read an arbitrarily large body into memory. Checks the
    Content-Length header first (cheap), then counts bytes for chunked uploads that omit it.
    """

    def __init__(self, app, max_bytes: int = 16 * 1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        declared = dict(scope.get("headers", [])).get(b"content-length")
        try:
            if declared is not None and int(declared) > self.max_bytes:
                return await self._reject(send)
        except ValueError:
            return await self._reject(send)             # a nonsense Content-Length is never valid

        received, started = 0, False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started:
                await self._reject(send)

    @staticmethod
    async def _reject(send):
        body = b'{"error":{"code":"payload_too_large","message":"Request body too large."}}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})
