// Consumes GET /api/rag/answer (Server-Sent Events, JSON payloads).
// EventSource reconnects when a stream ends, which would re-run (and re-bill) the question:
// every terminal path below closes it.
(() => {
  const form = document.getElementById("ask");
  const input = document.getElementById("question");
  const status = document.getElementById("status");
  const answerBox = document.getElementById("answer");
  const hitsList = document.getElementById("hits");
  const dialog = document.getElementById("article-dialog");
  let source = null;

  const setBusy = (busy) => { form.querySelector("button").setAttribute("aria-busy", busy); };

  function renderAnswer(text, citations) {
    answerBox.replaceChildren();
    for (const part of text.split(/(\[art:\d+\])/)) {
      const m = part.match(/^\[art:(\d+)\]$/);
      if (m) {
        const link = document.createElement("a");
        link.href = "#";
        link.textContent = part;
        link.dataset.article = m[1];
        answerBox.append(link);
      } else {
        answerBox.append(document.createTextNode(part));
      }
    }
    for (const li of hitsList.children) {
      li.classList.toggle("cited", citations.includes(Number(li.dataset.article)));
    }
  }

  function renderHits(hits) {
    hitsList.replaceChildren(...hits.map((h) => {
      const li = document.createElement("li");
      li.dataset.article = h.id;
      const link = document.createElement("a");
      link.href = "#";
      link.dataset.article = h.id;
      link.textContent = `[art:${h.id}]`;
      const small = document.createElement("small");
      small.textContent = h.source_act;
      li.append(link, ` ${h.path} `, small);
      return li;
    }));
  }

  function finish(message) {
    if (source) { source.close(); source = null; }
    setBusy(false);
    status.textContent = message;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || source) return;
    answerBox.replaceChildren();
    hitsList.replaceChildren();
    status.textContent = "Ricerca degli articoli…";
    setBusy(true);
    source = new EventSource(`/api/rag/answer?q=${encodeURIComponent(q)}`);
    source.addEventListener("hits", (ev) => {
      renderHits(JSON.parse(ev.data));
      status.textContent = "Generazione della risposta…";
    });
    source.addEventListener("answer", (ev) => {
      const a = JSON.parse(ev.data);
      renderAnswer(a.text, a.citations);
      let note = a.truncated ? "Risposta troncata (limite di lunghezza)." : "";
      if (a.invalid_citations.length) note += ` Citazioni non verificate: ${a.invalid_citations.join(", ")}.`;
      finish(note.trim());
    });
    source.addEventListener("error", (ev) => {
      if (ev.data) { finish(JSON.parse(ev.data).message); return; }  // server-sent error event
      // transport error: HTTP 401/422/429 or network. EventSource hides the status.
      finish("Richiesta non riuscita.");
      fetch("/api/config").then((r) => {
        if (r.status === 401) window.location = "/login";
        else status.textContent = "Richiesta non riuscita: quota giornaliera esaurita o domanda troppo lunga.";
      });
    });
  });

  document.addEventListener("click", async (e) => {
    const target = e.target.closest("[data-article]");
    if (target && target.tagName === "A") {
      e.preventDefault();
      const r = await fetch(`/api/articles/${target.dataset.article}`);
      if (!r.ok) return;
      const art = await r.json();
      document.getElementById("article-title").textContent = `[art:${art.id}] ${art.path}`;
      document.getElementById("article-text").textContent = `${art.source_act}\n\n${art.text}`;
      dialog.showModal();
    }
    if (e.target.matches("[data-close]")) dialog.close();
  });
})();
