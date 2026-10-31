"""Step 11: the HTTP API. Thin on purpose - every rule lives in serving/api_core.py (tested
without a web framework). Routes here only translate HTTP to those calls.

    uvicorn serving.app:create_app --factory --port 8000         (or: python scripts/run_api.py)

Run ONE worker: the embedding model and LLM client live in this process's memory, and the
generation gate that protects the CPU is per-process.
"""
from __future__ import annotations

import hmac
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from serving.api_core import ApiError, ApiService, Principal, sse
from serving.http_metrics import HttpMetrics
from serving.http_security import BodyLimit, SecurityHeaders

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("rag.api")


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)      # hard cap; the service enforces the friendlier limit


def create_app(service: ApiService | None = None, metrics_token: str | None = None,
               docs: bool | None = None) -> FastAPI:
    """`docs` controls the interactive API docs (/docs, /redoc, /openapi.json). Default: on in
    development, OFF when APP_ENV=production (they describe every route to anyone who asks)."""
    if service is None:
        from core.settings import settings
        from serving.bootstrap import build_api_service

        service = build_api_service()
        if metrics_token is None:
            metrics_token = settings.metrics_token
        if docs is None:
            docs = settings.app_env != "production"
    docs = True if docs is None else docs
    svc = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if svc.warm_up:       # load the model in the background so the first question isn't slow
            def warm():
                try:
                    svc.warm_up()
                    log.info("LLM warmed up")
                except Exception as exc:
                    log.warning("LLM warm-up failed: %s", exc)

            threading.Thread(target=warm, daemon=True, name="llm-warmup").start()
        yield

    app = FastAPI(title="Enterprise Knowledge Retrieval Platform", version="1.0.0", lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None,
                  openapi_url="/openapi.json" if docs else None)
    app.add_middleware(BodyLimit, max_bytes=16 * 1024)
    app.add_middleware(SecurityHeaders, frame_origin=(svc.grafana or {}).get("origin", ""))
    app.add_middleware(HttpMetrics, metrics=svc.metrics)      # outermost: also times the headers layer
    bearer = HTTPBearer(auto_error=False)

    def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
        return svc.authenticate(creds.credentials if creds else None)

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, headers=exc.headers,
                            content={"error": {"code": exc.code, "message": exc.message}})

    # ---- plain `def` routes run in a worker thread, which is what we want: the work blocks ----
    @app.post("/auth/login")
    def login(body: LoginRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        return svc.login(body.username, body.password, ip)

    @app.get("/me")
    def me(user: Principal = Depends(current_user)):
        return {"username": user.username, "role": user.role}

    @app.post("/ask")
    def ask(body: AskRequest, user: Principal = Depends(current_user)):
        return svc.ask(user, body.question)

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, user: Principal = Depends(current_user)):
        events = svc.ask_stream(user, body.question)     # HTTP-level failures raise here, pre-stream
        return StreamingResponse(
            (sse(e) for e in events), media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/admin/summary")
    def admin_summary(user: Principal = Depends(current_user)):
        return svc.admin_summary(user)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        checks = svc.readiness()
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse(status_code=200 if ok else 503,
                            content={"status": "ready" if ok else "degraded", "checks": checks})

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        """Prometheus scrape target. Aggregate numbers only (no usernames, no question text).
        If METRICS_TOKEN is set, scrapers must send it as a bearer token."""
        if metrics_token:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {metrics_token}".encode()):
                return JSONResponse(status_code=401, content={"error": {"code": "unauthorized",
                                                                         "message": "Bad metrics token."}})
        body, content_type = svc.metrics.render()
        return Response(content=body, media_type=content_type, headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/admin", include_in_schema=False)
    def admin_page():
        """Only a shell; every number comes from /admin/summary, which checks the role."""
        return FileResponse(STATIC / "admin.html", headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app
