# Security notes

What is protected, how, and what is **not** covered. Read the last section before deploying
anywhere real.

## Controls in place

| Concern | Control | Where | Tested by |
|---|---|---|---|
| Passwords | salted scrypt; verification always runs (dummy hash) so unknown users cost the same time | `serving/security.py` | `test_security.py` |
| Sessions | HS256 JWT carrying only the username; role looked up on every request, so disabling an account or changing a role is immediate | `serving/api_core.py` | `test_api_core.py` |
| Weak secrets | `APP_ENV=production` refuses to start with the default or a short `JWT_SECRET` | `serving/security.py` | `test_security.py` |
| Brute force | per-username and per-IP login limits; per-user question limit; Redis-backed when available (in-process otherwise; if Redis fails mid-run the limiter fails open rather than locking everyone out) | `serving/ratelimit.py` | `test_ratelimit_cache_gate.py` |
| Data leakage between roles | permission filter inside every vector query; cache keyed on the role's permission fingerprint; unknown role = error | `serving/retrieval.py`, `cache.py` | `test_retrieval.py`, `test_api_core.py`, `scripts/run_eval.py` (0 leaks) |
| Prompt injection | sources are wrapped as data and the model is told to ignore instructions inside them; `<source>` tags in content are neutralised; the system prompt is checked for leakage in evaluation | `serving/generation.py`, `eval/answer_eval.py` | `test_generation.py`, golden-set injection cases |
| Unsupported answers | citations must point at real sources; uncited answers are withheld | `serving/generation.py` | `test_generation.py` |
| Overload | one concurrent generation, a small queue, then HTTP 503 | `serving/concurrency.py` | `test_ratelimit_cache_gate.py` |
| Large requests | bodies over 16 KB are refused with 413 before being read | `serving/http_security.py` | `test_http_and_ui.py` |
| Browser attacks | strict CSP (no inline script), `X-Frame-Options: DENY`, `nosniff`, no-referrer; the chat page renders text only (never `innerHTML`); HSTS when served over HTTPS | `serving/http_security.py`, `static/` | `test_http_and_ui.py` |
| Information disclosure | API docs (`/docs`, `/openapi.json`) are off when `APP_ENV=production`; raw model output and internal errors are never in responses | `serving/app.py`, `api_core.py` | `test_app.py`, `test_api_core.py` |
| Logs | audit log records user, role, status, timing and a hash + length of the question, never its text | `serving/api_core.py` | `test_api_core.py` |
| Admin page | `/admin/summary` requires an admin role (checked in `ApiService.admin_summary`); CSP `frame-src` allows only the configured Grafana origin and only on `/admin`; aggregate numbers only | `serving/api_core.py`, `serving/http_security.py` | `test_admin.py` |
| Metrics | aggregate only; no usernames, questions or raw URLs as labels; optional bearer token | `serving/metrics.py` | `test_metrics.py` |
| Deletion safety | sync never deletes on a failed/partial listing; mass deletion needs an explicit flag | `ingestion/sync.py` | `test_sync.py` |

## Not covered (be honest about these)

- **TLS.** The app speaks plain HTTP. Put it behind a reverse proxy that terminates TLS
  (and set `--host 0.0.0.0` only there). Without TLS, tokens and passwords cross the network in clear text.
- **Single sign-on.** Users are local accounts in Postgres. A real deployment would use the
  company's identity provider (OIDC/SAML) and take roles from group membership.
- **Client IP behind a proxy.** The login IP limit uses the socket address; behind a proxy every
  user would share it. Configure the proxy to pass the real IP and teach the app to trust it.
- **Dev defaults.** `docker-compose.yml` publishes Postgres, Qdrant, Redis, MinIO, Prometheus and
  Grafana on all host interfaces with development passwords. Fine on a laptop; on any shared
  network bind them to `127.0.0.1` or put them on a private network, and change every password.
  The Airflow setup uses the default auth manager and default secrets.
- **Secrets storage.** `.env` is plain text. Keep it out of cloud-synced folders (OneDrive) and out
  of git (it is in `.gitignore`); use a secrets manager in production. Rotate the seeded user
  passwords and delete any file that holds them.
- **Token revocation list / refresh tokens / MFA.** Tokens last `JWT_EXPIRE_MINUTES` (60); the
  live role lookup covers disabling a user, nothing more.
- **Request-rate limits at the edge.** Per-user limits exist; volumetric DoS protection is the
  proxy's job.
- **Data governance.** No retention policy for audit logs, no encryption at rest configuration for
  Postgres/MinIO/Qdrant volumes, no PII detection in indexed documents.
- **Model supply chain.** Models are downloaded from public hubs on first use; pin versions and
  mirror them for a locked-down environment.
- **Adversarial prompt injection.** The defences above raise the bar; they do not make a 3B model
  injection-proof. Treat the model's text as untrusted output (the UI already does).

## Reporting / hardening checklist before a real deployment

1. TLS in front; real SSO; strong `JWT_SECRET`; `APP_ENV=production`.
2. Private network for the data stores; new passwords; `METRICS_TOKEN` set.
3. Backups running and a restore rehearsed (`docs/OPERATIONS.md`).
4. `python scripts/run_eval.py --gate` passing on the real corpus with a reviewed golden set.
5. Permission mapping from the real source's ACLs reviewed with the data owner.

## Admin page and embedded Grafana

The embedded dashboard works because the development compose file enables anonymous **viewer**
access and embedding in Grafana. That means anyone who can reach port 3000 can read the dashboards
(aggregate numbers, no user data) even without opening `/admin`. For a shared or production
deployment: remove `GF_AUTH_ANONYMOUS_*`, put Grafana behind the reverse proxy with SSO or an
auth proxy, and keep `GF_SECURITY_ALLOW_EMBEDDING` only for that proxy origin. The native live
numbers on `/admin` do not depend on Grafana.
