"""Pure ASGI middleware that records request counts and durations per ROUTE TEMPLATE
(e.g. "/ask/stream"), never per raw URL, so label cardinality stays fixed.

Duration is measured until the response has been fully sent, which for the streaming answer
endpoint is the whole answer.
"""
from __future__ import annotations

import time

SKIP = ("/metrics",)          # scraping the metrics must not move the metrics


class HttpMetrics:
    def __init__(self, app, metrics, skip: tuple[str, ...] = SKIP):
        self.app, self.metrics, self.skip = app, metrics, skip

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        began = time.perf_counter()
        status = {"code": 500}                    # what a client sees if the app crashes mid-way

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")            # set by the router once it has matched
            path = getattr(route, "path", None) or "unmatched"
            if path not in self.skip:
                self.metrics.http(scope.get("method", "GET"), path, status["code"],
                                  time.perf_counter() - began)
