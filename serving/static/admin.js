/* Admin page. Reads the same session token as the chat page; the server decides who may see data. */
(function () {
  "use strict";
  const TOKEN_KEY = "ekrp_token";
  const $ = (id) => document.getElementById(id);
  let token = null;
  try { token = sessionStorage.getItem(TOKEN_KEY); } catch (e) {}

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function fail(message) {
    const box = $("admin-error");
    box.textContent = message; box.hidden = false;
    $("admin-view").hidden = true;
  }
  const fmtSeconds = (v) => (v === null || v === undefined) ? "n/a" : (v < 1 ? Math.round(v * 1000) + " ms" : v.toFixed(1) + " s");
  const fmtPercent = (v) => (v === null || v === undefined) ? "n/a" : Math.round(v * 100) + "%";
  function fmtAge(seconds) {
    if (seconds === null || seconds === undefined) return "never";
    if (seconds < 3600) return Math.round(seconds / 60) + " min ago";
    if (seconds < 172800) return Math.round(seconds / 3600) + " h ago";
    return Math.round(seconds / 86400) + " days ago";
  }

  function tile(label, value, tone) {
    const t = el("div", "tile" + (tone ? " " + tone : ""));
    t.appendChild(el("div", "tile-value", String(value)));
    t.appendChild(el("div", "tile-label", label));
    return t;
  }
  function fillList(id, rows, empty) {
    const ul = $(id); ul.textContent = "";
    if (!rows.length) { ul.appendChild(el("li", "muted", empty)); return; }
    for (const [name, value, tone] of rows) {
      const li = el("li");
      li.appendChild(el("span", null, name));
      li.appendChild(el("strong", tone || "", String(value)));
      ul.appendChild(li);
    }
  }

  function render(data) {
    const m = data.metrics || {};
    const asks = m.asks || {}, lat = m.latency_seconds || {}, cache = m.cache || {};
    const gen = m.generation || {}, http = m.http || {};
    const tiles = $("tiles"); tiles.textContent = "";
    tiles.appendChild(tile("Questions", asks.total || 0));
    tiles.appendChild(tile("Answer p50", fmtSeconds(lat.p50)));
    tiles.appendChild(tile("Answer p95", fmtSeconds(lat.p95)));
    tiles.appendChild(tile("Cache hit rate", fmtPercent(cache.hit_ratio)));
    tiles.appendChild(tile("Server errors", http.server_errors || 0, (http.server_errors || 0) > 0 ? "bad" : ""));
    tiles.appendChild(tile("Generating now", (gen.in_flight ?? "n/a") + " / " + (gen.capacity ?? "n/a")));
    tiles.appendChild(tile("Waiting", gen.waiting ?? "n/a"));
    tiles.appendChild(tile("Last sync", fmtAge(m.last_sync_age_seconds)));

    const deps = Object.entries(m.dependencies || {}).map(
      ([n, up]) => [n, up ? "up" : "DOWN", up ? "ok" : "bad"]);
    fillList("deps", deps, "No dependency data yet.");
    const c = data.controls || {};
    fillList("controls", [
      ["Questions", c.ask_limit], ["Logins", c.login_limit], ["Logins per IP", c.ip_login_limit],
      ["Rate-limit store", c.limiter_backend], ["Answer cache", c.cache_backend],
      ["Concurrent generations", c.max_concurrent_generations],
    ].filter((r) => r[1] !== undefined && r[1] !== null), "");
    fillList("outcomes", Object.entries(asks.by_status || {}), "No questions yet.");
    const turned = Object.entries(m.rejections || {}).map(([k, v]) => [k, v]);
    for (const [k, v] of Object.entries(m.logins || {})) if (k !== "ok") turned.push(["login " + k, v]);
    fillList("rejections", turned, "Nothing turned away.");

    const note = $("grafana-note"), holder = $("grafana-frame");
    holder.textContent = "";
    if (data.grafana && data.grafana.dashboard_url) {
      note.textContent = "Time series, alerts and history. Opens read-only; log in to Grafana itself to edit.";
      const frame = document.createElement("iframe");
      frame.src = data.grafana.dashboard_url;
      frame.title = "Grafana dashboard";
      frame.loading = "lazy";
      frame.referrerPolicy = "no-referrer";
      holder.appendChild(frame);
    } else {
      note.textContent = "Grafana is not configured (set GRAFANA_URL). The live numbers above still work.";
    }
    $("updated").textContent = "Updated " + new Date().toLocaleTimeString();
    $("admin-view").hidden = false;
    $("admin-error").hidden = true;
  }

  async function load() {
    if (!token) return fail("Please sign in on the chat page first, then open Admin.");
    try {
      const response = await fetch("/admin/summary", { headers: { Authorization: "Bearer " + token } });
      if (response.status === 401) return fail("Your session expired. Sign in on the chat page again.");
      if (response.status === 403) return fail("This page is for administrators.");
      if (!response.ok) return fail("Could not load the summary (" + response.status + ").");
      const data = await response.json();
      render(data);
    } catch (e) { fail("Could not reach the server."); }
  }

  $("refresh").addEventListener("click", load);
  load();
  setInterval(() => { if (!document.hidden && !$("admin-view").hidden) load(); }, 15000);
})();
