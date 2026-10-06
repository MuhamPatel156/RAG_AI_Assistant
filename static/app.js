"use strict";

const $ = (id) => document.getElementById(id);
const state = { document: null, configured: false, uploading: false, asking: false, timer: null };

function remember(id) {
  try { id ? sessionStorage.setItem("self-rag-document", id) : sessionStorage.removeItem("self-rag-document"); }
  catch { /* The app also works when browser storage is disabled. */ }
}

async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error("Cannot reach the server. Check that it is running, then try again."); }
  let data;
  try { data = await response.json(); }
  catch { throw new Error("The server returned an unexpected response. Please try again."); }
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : "Please check your input and try again.";
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return data;
}

function showError(id, message = "") {
  $(id).textContent = message;
  $(id).hidden = !message;
}

function syncControls() {
  const doc = state.document;
  const ready = doc?.status === "ready" && state.configured;
  $("upload-button").disabled = !state.configured || state.uploading || Boolean(doc);
  $("dropzone").hidden = Boolean(doc);
  $("document-card").hidden = !doc;
  $("source-count").textContent = doc ? "01" : "00";
  $("question-input").disabled = !ready || state.asking;
  $("ask-button").disabled = !ready || state.asking || !$("question-input").value.trim();
  $("remove-button").disabled = state.asking || doc?.status === "processing";
  $("clear-button").disabled = state.asking;
  document.querySelectorAll(".suggestion").forEach((button) => { button.disabled = state.asking; });
  $("composer-hint").textContent = state.asking ? "Reading and reviewing your question…" : ready ? doc.filename : doc ? "Preparing your document…" : "Upload a document to start";
  if (doc?.status === "error") $("composer-hint").textContent = "Remove this document to try again";
  if (doc) {
    $("document-name").textContent = doc.filename;
    $("document-meta").textContent = ready ? `${doc.pages} ${doc.pages === 1 ? "page" : "pages"} · ${doc.chunks} searchable passages` : "Your document workspace";
    $("document-status").textContent = doc.message;
    $("document-indicator").className = `status-dot${doc.status === "processing" ? " working" : ""}`;
  }
}

async function checkHealth() {
  try {
    const health = await api("/api/health");
    state.configured = health.configured;
    $("connection").className = `connection${health.configured ? "" : " warning"}`;
    $("connection-text").textContent = health.configured ? "Ready to connect" : "Setup required";
    $("setup-notice").hidden = health.configured;
    $("setup-text").textContent = `Add ${health.missing_keys.join(" and ")} to the server's .env file, then restart it. Your keys stay on the backend.`;
  } catch (error) {
    state.configured = false;
    $("connection").className = "connection offline";
    $("connection-text").textContent = "Server offline";
    $("setup-notice").hidden = false;
    $("setup-text").textContent = error.message;
  }
  syncControls();
}

async function pollDocument() {
  clearTimeout(state.timer);
  if (!state.document) return;
  try {
    state.document = await api(`/api/documents/${state.document.id}`);
    syncControls();
    showError("upload-error");
    if (state.document.status === "processing") state.timer = setTimeout(pollDocument, 1500);
    else if (state.document.status === "ready") {
      $("connection-text").textContent = "Document ready";
      $("question-input").focus({ preventScroll: true });
    }
  } catch (error) {
    showError("upload-error", error.message);
    if (error.status === 404) { state.document = null; remember(null); }
    else state.timer = setTimeout(pollDocument, 5000);
    syncControls();
  }
}

async function uploadFile(file) {
  if (!file || state.uploading || state.document || !state.configured) return;
  showError("upload-error");
  if (!file.name.toLowerCase().endsWith(".pdf")) return showError("upload-error", "Please choose a PDF file.");
  if (file.size > 20 * 1024 * 1024) return showError("upload-error", "PDF files must be 20 MB or smaller.");
  if (file.size === 0) return showError("upload-error", "This file is empty. Please choose another PDF.");
  state.uploading = true;
  $("upload-button").textContent = "Uploading…";
  syncControls();
  try {
    const body = new FormData();
    body.append("file", file);
    state.document = await api("/api/documents", { method: "POST", body });
    remember(state.document.id);
    await pollDocument();
  } catch (error) { showError("upload-error", error.message); }
  finally {
    state.uploading = false;
    $("upload-button").textContent = "Choose a PDF +";
    $("file-input").value = "";
    syncControls();
  }
}

function clearConversation() {
  $("conversation").replaceChildren();
  $("conversation").hidden = true;
  $("empty-state").hidden = false;
  $("clear-button").hidden = true;
  showError("question-error");
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderResult(card, data) {
  card.replaceChildren(el("div", "answer-label", "Self RAG"), el("div", "answer-text", data.answer));
  const meta = el("div", "answer-meta");
  meta.append(el("span", "", data.supported ? "Review complete" : "No supported answer"), el("span", "", `${data.elapsed_seconds}s`));
  if (data.revisions) meta.append(el("span", "", `${data.revisions} ${data.revisions === 1 ? "revision" : "revisions"}`));
  if (data.mode === "overview") meta.append(el("span", "", "Document overview"));
  card.append(meta);
  if (data.context_note) card.append(el("p", "pending-note", data.context_note));
  if (data.sources.length) {
    const details = el("details", "sources");
    details.append(el("summary", "", `${data.sources.length} retrieved passages · View sources`));
    const grid = el("div", "source-grid");
    for (const source of data.sources) {
      const item = el("div", "source-card");
      item.append(el("h3", "", `Source ${source.id} · ${source.filename} · Page ${source.page}`), el("p", "", source.excerpt));
      grid.append(item);
    }
    details.append(grid);
    card.append(details);
  }
  const review = el("details", "review");
  review.append(el("summary", "", "How this answer was reviewed"));
  const steps = el("ol");
  for (const step of data.steps) steps.append(el("li", "", step));
  review.append(steps);
  card.append(review);
}

async function askQuestion(event) {
  event.preventDefault();
  const question = $("question-input").value.trim();
  if (!question || state.asking || state.document?.status !== "ready") return;
  state.asking = true;
  showError("question-error");
  syncControls();
  $("empty-state").hidden = true;
  $("conversation").hidden = false;
  $("clear-button").hidden = false;
  const turn = el("article", "turn");
  const prompt = el("div", "user-question");
  prompt.append(el("span", "user-avatar", "YOU"), el("p", "question-text", question));
  const card = el("div", "answer-card");
  card.setAttribute("aria-busy", "true");
  const pending = el("div", "pending");
  pending.append(el("span", "spinner"), el("span", "", "Finding context and reviewing the answer…"));
  card.append(el("div", "answer-label", "Self RAG"), pending, el("p", "pending-note", "The workflow may take a minute, especially if the answer needs a revision."));
  turn.append(prompt, card);
  $("conversation").append(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const result = await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: state.document.id, question }),
    });
    renderResult(card, result);
    $("question-input").value = "";
  } catch (error) {
    card.replaceChildren(el("div", "answer-label", "Request couldn't be completed"), el("p", "answer-text failed-answer", error.message));
    showError("question-error", "Your question is still below. You can send it again.");
    if (error.status === 404) {
      state.document = null;
      remember(null);
      showError("upload-error", "The server restarted or this document was removed. Upload the PDF again.");
    }
  } finally {
    card.setAttribute("aria-busy", "false");
    state.asking = false;
    syncControls();
    $("question-input").focus({ preventScroll: true });
  }
}

$("upload-button").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => uploadFile(event.target.files[0]));
$("question-form").addEventListener("submit", askQuestion);
$("question-input").addEventListener("input", syncControls);
$("question-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("ask-button").disabled) $("question-form").requestSubmit();
  }
});
$("retry-connection").addEventListener("click", checkHealth);
$("clear-button").addEventListener("click", clearConversation);
$("remove-button").addEventListener("click", async () => {
  if (!state.document || state.asking) return;
  $("remove-button").disabled = true;
  try {
    await api(`/api/documents/${state.document.id}`, { method: "DELETE" });
    clearTimeout(state.timer);
    state.document = null;
    remember(null);
    $("question-input").value = "";
    clearConversation();
    showError("upload-error");
    $("connection-text").textContent = "Ready to connect";
  } catch (error) { showError("upload-error", error.message); }
  syncControls();
});

for (const name of ["dragenter", "dragover"]) $("dropzone").addEventListener(name, (event) => {
  event.preventDefault();
  if (state.configured) $("dropzone").classList.add("dragging");
});
for (const name of ["dragleave", "drop"]) $("dropzone").addEventListener(name, (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("dragging");
});
$("dropzone").addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => {
  $("question-input").value = button.dataset.question;
  syncControls();
  if (!$("question-input").disabled) $("question-input").focus();
  else if (!state.document) showError("upload-error", "Upload your PDF first. Your suggested question is saved below.");
}));

(async function init() {
  await checkHealth();
  let saved;
  try { saved = sessionStorage.getItem("self-rag-document"); } catch { /* Storage is optional. */ }
  if (saved) { state.document = { id: saved, status: "processing", filename: "Restoring document…", message: "Checking the server…" }; syncControls(); await pollDocument(); }
})();
