/* Pure helpers (no DOM), kept separate so they can be unit-tested with Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.EKRP = factory();
})(typeof self !== "undefined" ? self : this, function () {
  /* Server-Sent Events arrive in arbitrary chunks. Feed the text received so far; get back the
     complete events plus whatever is left over (an unfinished frame) to prepend next time. */
  function parseSSE(buffer) {
    const normalized = buffer.replace(/\r\n/g, "\n");
    const frames = normalized.split("\n\n");
    const rest = frames.pop();
    const events = [];
    for (const frame of frames) {
      let name = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      try { events.push({ event: name, data: JSON.parse(data.join("\n")) }); }
      catch (e) { /* ignore a malformed frame rather than breaking the stream */ }
    }
    return { events, rest };
  }

  /* "Do X [1]. Then Y [2, 3]." -> text and citation parts, so the page can render the markers
     as chips WITHOUT ever treating model output as HTML. */
  function splitCitations(text) {
    const parts = [];
    const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
    let last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) parts.push({ type: "text", value: text.slice(last, m.index) });
      parts.push({ type: "cite", nums: m[1].split(",").map((n) => parseInt(n.trim(), 10)) });
      last = re.lastIndex;
    }
    if (last < text.length) parts.push({ type: "text", value: text.slice(last) });
    return parts;
  }

  return { parseSSE, splitCitations };
});
