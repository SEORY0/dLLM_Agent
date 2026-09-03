import { $, escapeHtml } from "./ui.js";

export function renderEvidence(report) {
  const fuzz = report?.fuzz;
  const rows = fuzz ? [...(fuzz.corpus_artifacts || []), ...(fuzz.crash_artifacts || [])] : [];
  $("[data-role='evidence-body']").innerHTML = rows.length ? rows.map((artifact) => (
    `<tr><td>${escapeHtml(artifact.name)}</td><td>${escapeHtml(fuzz.oracle_status || "stored")}</td>`
    + `<td>${artifact.sha256.slice(0, 14)}</td><td>${artifact.size}</td>`
    + `<td><button class="copy-button" type="button" data-copy-hash="${artifact.sha256}" `
    + `aria-label="Copy SHA-256 for ${escapeHtml(artifact.name)}" aria-live="polite">Copy</button></td></tr>`
  )).join("") : `<tr><td colspan="5">No report artifacts loaded from the API.</td></tr>`;
}

export async function copyEvidenceHash(event) {
  const target = event.target instanceof Element ? event.target.closest("[data-copy-hash]") : null;
  if (!(target instanceof HTMLButtonElement)) return;
  try {
    await navigator.clipboard.writeText(target.dataset.copyHash);
    target.textContent = "Copied";
  } catch {
    target.textContent = "Unavailable";
  }
}
