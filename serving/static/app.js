/* Chat UI. All model and server text is inserted as plain text nodes, never parsed as markup. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const TOKEN_KEY = "ekrp_token";
  let token = null, user = null;
  try { token = sessionStorage.getItem(TOKEN_KEY); } catch (e) { /* storage blocked: stay in memory */ }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function show(view) {
    $("login-view").hidden = view !== "login";
    $("chat-view").hidden = view !== "chat";
    $("whoami").hidden = view !== "chat";
  }

  function logout(message) {
    token = null; user = null;
    try { sessionStorage.removeItem(TOKEN_KEY); } catch (e) {}
    show("login");
    const err = $("login-error");
    err.textContent = message || ""; err.hidden = !message;
  }

  async function errorFrom(response) {
    let message = "Something went wrong (" + response.status + ").";
    try { const body = await response.json(); if (body.error) message = body.error.message; } catch (e) {}
    const wait = response.headers.get("Retry-After");
    if (wait) message += " (try again in " + wait + "s)";
    return message;
  }

  /* ---------------------------------------------------------------- login */
  $("login-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const err = $("login-error"); err.hidden = true;
    try {
      const response = await fetch("/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: $("username").value, password: $("password").value }),
      });
      if (!response.ok) { err.textContent = await errorFrom(response); err.hidden = false; return; }
      const body = await response.json();
      token = body.access_token; user = { username: body.username, role: body.role };
      try { sessionStorage.setItem(TOKEN_KEY, token); } catch (e) {}
      $("password").value = "";
      enterChat();
    } catch (e) { err.textContent = "Could not reach the server."; err.hidden = false; }
  });
  $("logout").addEventListener("click", () => logout(""));

  async function enterChat() {
    if (!user) {                                   // page reload: recover who we are from the token
      try {
        const r = await fetch("/me", { headers: { Authorization: "Bearer " + token } });
        if (!r.ok) return logout("");
        user = await r.json();
      } catch (e) { return logout("Could not reach the server."); }
    }
    $("user-chip").textContent = user.username + " · " + user.role;
    $("admin-link").hidden = user.role !== "admin";      // convenience only; /admin/summary enforces it
    show("chat");
    $("question").focus();
  }

  /* ----------------------------------------------------------------- chat */
  function renderSources(container, sources, heading, flashTargets) {
    container.textContent = "";
    if (!sources.length) return;
    container.appendChild(el("h3", null, heading));
    for (const s of sources) {
      const li = el("li");
      li.dataset.n = s.n;
      const title = el("div", "src-title");
      title.appendChild(el("span", "n", "[" + s.n + "]"));
      title.appendChild(document.createTextNode(s.section ? s.title + " › " + s.section : s.title));
      li.append(title, el("div", "src-path", s.path), el("div", "src-snippet", s.snippet));
      container.appendChild(li);
    }
  }

  function renderAnswerText(box, text, listEl) {
    box.textContent = "";
    for (const part of EKRP.splitCitations(text)) {
      if (part.type === "text") { box.appendChild(document.createTextNode(part.value)); continue; }
      for (const n of part.nums) {
        const chip = el("button", "cite", String(n));
        chip.type = "button"; chip.title = "Show source " + n;
        chip.addEventListener("click", () => {
          const li = listEl.querySelector('li[data-n="' + n + '"]');
          if (li) { li.scrollIntoView({ block: "nearest", behavior: "smooth" }); li.classList.add("flash");
                    setTimeout(() => li.classList.remove("flash"), 1500); }
        });
        box.appendChild(chip);
      }
    }
  }

  function addTurn(question) {
    const turn = el("div", "turn");
    turn.appendChild(el("div", "q", question));
    const a = el("div", "a");
    const status = el("div", "status", "Searching the handbook…");
    const text = el("div", "text draft");
    const list = el("ul", "sources");
    a.append(status, text, list);
    turn.appendChild(a);
    $("transcript").querySelector(".hint")?.remove();
    $("transcript").appendChild(turn);
    turn.scrollIntoView({ block: "end", behavior: "smooth" });
    return { status, text, list, turn };
  }

  function finish(ui, answer) {
    ui.status.textContent = "";
    ui.status.appendChild(el("span", "badge " + answer.status, answer.status.replace("_", " ")));
    if (answer.cached) ui.status.appendChild(el("span", "badge", "cached"));
    ui.text.className = "text";
    renderAnswerText(ui.text, answer.text, ui.list);
    if (answer.citations.length) renderSources(ui.list, answer.citations, "Sources", true);
    else if (answer.related.length) renderSources(ui.list, answer.related, "Related pages", false);
    else ui.list.textContent = "";
    const ms = answer.timings_ms && answer.timings_ms.total_ms;
    if (ms) ui.turn.querySelector(".a").appendChild(el("div", "meta", (ms / 1000).toFixed(1) + " s"));
  }

  $("ask-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const question = $("question").value.trim();
    if (!question) return;
    const button = $("ask-button");
    button.disabled = true; $("question").value = "";
    const ui = addTurn(question);
    try {
      const response = await fetch("/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
        body: JSON.stringify({ question }),
      });
      if (response.status === 401) return logout("Your session expired. Please sign in again.");
      if (!response.ok) { ui.status.textContent = await errorFrom(response); ui.text.textContent = ""; return; }

      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = "", draft = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = EKRP.parseSSE(buffer); buffer = parsed.rest;
        for (const { event, data } of parsed.events) {
          if (event === "sources") { ui.status.textContent = "Writing an answer…"; renderSources(ui.list, data.sources, "Retrieved pages", false); }
          else if (event === "token") { draft += data.text; ui.text.textContent = draft; }
          else if (event === "done") finish(ui, data.answer);
          else if (event === "error") { ui.status.textContent = data.message; ui.text.textContent = ""; }
        }
      }
    } catch (e) {
      ui.status.textContent = "Connection lost. Please try again.";
    } finally { button.disabled = false; $("question").focus(); }
  });

  $("question").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("ask-form").requestSubmit(); }
  });

  if (token) enterChat(); else show("login");
})();
