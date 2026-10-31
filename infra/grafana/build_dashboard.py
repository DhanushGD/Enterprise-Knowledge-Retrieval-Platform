"""Generates infra/grafana/dashboards/ekrp-overview.json (a test fails if the file drifts).

    python infra/grafana/build_dashboard.py

Edit panels here, regenerate, commit both. Grafana reloads the file within ~30 s.
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).parent / "dashboards" / "ekrp-overview.json"
_next_id = [0]


def _panel(kind, title, targets, x, y, w, h, unit=None, desc=None, extra=None):
    _next_id[0] += 1
    panel = {
        "id": _next_id[0], "type": kind, "title": title, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": expr, "legendFormat": legend}
                    for i, (expr, legend) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit} if unit else {}, "overrides": []},
    }
    if desc:
        panel["description"] = desc
    if kind == "stat":
        panel["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                            "colorMode": "value", "graphMode": "area", "textMode": "auto"}
    if extra:
        panel.update(extra)
    return panel


def stat(title, expr, x, w=4, unit=None, desc=None, thresholds=None):
    p = _panel("stat", title, [(expr, "")], x, 0, w, 4, unit, desc)
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
        p["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
    return p


def series(title, targets, x, y, w=12, h=8, unit=None, desc=None, stacked=False):
    p = _panel("timeseries", title, targets, x, y, w, h, unit, desc)
    if stacked:
        p["fieldConfig"]["defaults"]["custom"] = {"stacking": {"mode": "normal"}, "fillOpacity": 30}
    return p


def quantile(q, metric, extra=""):
    return f"histogram_quantile({q}, sum by (le) (rate({metric}_bucket{extra}[5m])))"


GREEN_RED = [{"color": "red", "value": None}, {"color": "green", "value": 1}]


def build() -> dict:
    _next_id[0] = 0
    fresh = '{cached="false"}'
    panels = [
        stat("Questions (1 h)", "sum(increase(ekrp_asks_total[1h]))", 0),
        stat("Answered share (1 h)",
             'sum(increase(ekrp_asks_total{status="answered"}[1h])) / sum(increase(ekrp_asks_total[1h]))',
             4, unit="percentunit", desc="Includes refused off-topic questions, so this is not an accuracy number."),
        stat("Cache hit ratio (1 h)",
             'sum(increase(ekrp_cache_total{result="hit"}[1h])) / sum(increase(ekrp_cache_total[1h]))',
             8, unit="percentunit"),
        stat("p95 answer time (15 m)", quantile(0.95, "ekrp_ask_seconds", fresh).replace("[5m]", "[15m]"),
             12, unit="s", thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 30},
                                       {"color": "red", "value": 60}]),
        stat("Knowledge base age", "(time() - max(ekrp_last_sync_timestamp_seconds)) / 3600", 16, unit="h",
             desc="Hours since the last ingestion run finished.",
             thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 26},
                         {"color": "red", "value": 36}]),
        stat("Dependencies up", "min(ekrp_dependency_up)", 20, desc="1 = postgres, qdrant, ollama (and redis) all answer.",
             thresholds=GREEN_RED),

        series("Questions per minute, by outcome",
               [('sum by (status) (rate(ekrp_asks_total[5m])) * 60', "{{status}}")], 0, 4, unit="short", stacked=True),
        series("Answer time, fresh answers (cache hits excluded)",
               [(quantile(0.5, "ekrp_ask_seconds", fresh), "p50"),
                (quantile(0.95, "ekrp_ask_seconds", fresh), "p95"),
                (quantile(0.99, "ekrp_ask_seconds", fresh), "p99")], 12, 4, unit="s"),

        series("Where the time goes (p95)",
               [(quantile(0.95, "ekrp_retrieval_seconds"), "retrieval"),
                (quantile(0.95, "ekrp_first_token_seconds"), "until first token"),
                (quantile(0.95, "ekrp_llm_seconds"), "model total")], 0, 12, unit="s"),
        series("Generation gate (the CPU)",
               [("ekrp_generation_in_flight", "generating"), ("ekrp_generation_waiting", "waiting"),
                ("ekrp_generation_capacity", "capacity")], 12, 12, unit="short",
               desc="Waiting above 0 means users are queueing for the model."),

        series("Requests turned away (per 5 min)",
               [('sum by (scope, code) (increase(ekrp_rejections_total[5m]))', "{{scope}} / {{code}}")],
               0, 20, unit="short", desc="rate_limited = a user asked too fast; busy = the model is saturated."),
        series("Logins (per 5 min)",
               [('sum by (outcome) (increase(ekrp_logins_total[5m]))', "{{outcome}}")], 12, 20, unit="short"),

        series("Why questions were refused (per minute)",
               [('sum by (reason) (rate(ekrp_asks_total{status="abstained"}[15m])) * 60', "{{reason}}")],
               0, 28, unit="short", stacked=True,
               desc="low_relevance = the relevance gate; model_declined = the model said it could not answer."),
        series("Cache lookups (per minute)",
               [('sum by (result) (rate(ekrp_cache_total[5m])) * 60', "{{result}}")], 12, 28, unit="short"),

        series("Documents by status",
               [("ekrp_documents", "{{status}}")], 0, 36, unit="short", stacked=True,
               desc="Anything ending in _failed is stuck and is retried by the next sync."),
        series("Model tokens per second",
               [('sum by (kind) (rate(ekrp_tokens_total[5m]))', "{{kind}}")], 12, 36, unit="short"),

        series("HTTP requests per minute",
               [('sum by (path, status) (rate(ekrp_http_requests_total[5m])) * 60', "{{path}} {{status}}")],
               0, 44, unit="short"),
        series("Server errors (5xx) and stream failures (per 10 min)",
               [('sum(increase(ekrp_http_requests_total{status=~"5.."}[10m]))', "5xx responses"),
                ('sum by (code) (increase(ekrp_stream_errors_total[10m]))', "stream {{code}}")],
               12, 44, unit="short"),
    ]
    return {
        "uid": "ekrp-overview", "title": "EKRP - Support assistant", "tags": ["ekrp", "rag"],
        "schemaVersion": 39, "version": 1, "editable": False, "timezone": "browser",
        "refresh": "30s", "time": {"from": "now-6h", "to": "now"},
        "panels": panels, "templating": {"list": []}, "annotations": {"list": []},
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
