import { observeReducedMotion } from "./motion.js";
import { $, api, escapeHtml, metric, percent, setText } from "./ui.js";
import { loadReport } from "./report-selection.js";
let state = { catalog: { resources: [] }, reports: {}, jobs: [], trajectory: null };
async function load() {
  if (document.body.dataset.page !== "console") return;
  observeReducedMotion((reduced) => { document.body.dataset.motion = reduced ? "reduced" : "ok"; });
  bindEvents();
  try {
    state.catalog = await api("/catalog");
    state.jobs = await loadJobs();
    await loadReports();
    setText("[data-bind='connection']", "API connected");
    setText("[data-role='catalog-status']", "Catalog loaded");
  } catch {
    setText("[data-bind='connection']", "API unavailable");
    setText("[data-role='catalog-status']", "Catalog unavailable");
  }
  await route();
}
function bindEvents() {
  document.addEventListener("click", copyHash);
}
async function loadJobs() {
  try {
    const response = await api("/jobs");
    return Array.isArray(response) ? response : response.jobs || [];
  } catch {
    return [];
  }
}
async function loadReports() {
  for (const resource of resources("hunt_report")) {
    state.reports[resource.resource_id] = await loadReport(api, state.reports, resource.resource_id);
  }
}
async function route() {
  const match = location.pathname.match(/^\/run\/([^/]+)$/);
  if (!match) {
    document.body.dataset.mode = "summary";
    $("[data-view='detail']").hidden = true;
    renderRuns();
    return;
  }
  document.body.dataset.mode = "detail";
  $("[data-view='detail']").hidden = false;
  await renderDetail(decodeURIComponent(match[1]));
}
function resources(kind) {
  return (state.catalog.resources || []).filter((item) => item.kind === kind);
}
function bankFor(report) {
  return resources("candidate_bank").find((bank) => bank.sha256 === report?.candidate_bank_sha256);
}
function jobStateFor(resource) {
  const path = resource?.relative_path || "";
  const job = state.jobs.find((item) => path.includes(item.job_id));
  return job?.state || "completed";
}
function renderRuns() {
  const rows = resources("hunt_report").map((resource) => runRow(resource, state.reports[resource.resource_id]));
  setText("[data-bind='run-count']", String(rows.length));
  setText("[data-bind='target']", rows.length ? "reports loaded" : "no reports");
  $("[data-role='runs-table-body']").innerHTML = rows.length
    ? rows.join("")
    : `<tr><td colspan="7">No hunt reports are available in the catalog.</td></tr>`;
}
function runRow(resource, report) {
  const bank = bankFor(report);
  const coverage = report?.coverage?.totals ? `lines ${percent(report.coverage.totals.lines)}` : "n/a";
  const artifacts = artifactCount(report);
  const findings = report?.fuzz?.oracle_status === "proved" ? "1 proved" : "0";
  const status = bank ? jobStateFor(resource) : "bank mismatch";
  return `<tr>
    <td><a href="/run/${encodeURIComponent(resource.resource_id)}">${escapeHtml(report?.hunt_id || resource.resource_id)}</a></td>
    <td><span class="badge ${statusClass(status)}">${escapeHtml(status)}</span></td>
    <td>${escapeHtml(targetName(report))}</td>
    <td>${report?.candidate_scores?.length || "n/a"}</td>
    <td>${coverage}</td>
    <td>${findings}</td>
    <td>${artifacts}</td>
  </tr>`;
}
async function renderDetail(resourceId) {
  const resource = resources("hunt_report").find((item) => item.resource_id === resourceId);
  const report = resource ? state.reports[resourceId] : null;
  const bank = bankFor(report);
  setText("[data-bind='run-count']", String(resources("hunt_report").length));
  setText("[data-role='detail-title']", report?.hunt_id || "Unknown run");
  setText("[data-bind='target']", targetName(report));
  if (!report || !bank) {
    renderMissingDetail(report);
    return;
  }
  try {
    state.trajectory = await api(`/trajectories/${encodeURIComponent(bank.resource_id)}`);
  } catch {
    state.trajectory = null;
  }
  renderDetailSummary(report);
  renderInstances(report, state.trajectory);
  renderCoverage(report.coverage?.totals);
  renderFuzz(report);
  renderEvidence(report);
  renderEvidenceDetails(report, state.trajectory);
}
function renderMissingDetail(report) {
  setText("[data-role='claim-copy']", report ? "Selected report does not match an available candidate bank; validation evidence is hidden." : "Unknown report resource.");
  $("[data-role='detail-metrics']").innerHTML = metric("Status", "unavailable");
  $("[data-role='instance-grid']").innerHTML = `<div class="empty-state">No matched candidate bank is available for this report.</div>`;
  $("[data-role='coverage-grid']").innerHTML = metric("Lines", "n/a") + metric("Functions", "n/a") + metric("Branches", "n/a") + metric("Regions", "n/a");
  $("[data-role='fuzz-grid']").innerHTML = metric("Oracle", "n/a");
  $("[data-role='evidence-body']").innerHTML = `<tr><td colspan="5">No artifacts rendered without a matched bank.</td></tr>`;
}
function renderDetailSummary(report) {
  const fuzz = report.fuzz || {};
  const selected = selectedCandidate(report, state.trajectory);
  const truth = trajectoryMode(selected);
  $("[data-role='detail-metrics']").innerHTML = [
    metric("Status", fuzz.oracle_status || "n/a"),
    metric("Target", targetName(report)),
    metric("Selected", report.selected_candidate_id),
    metric("Trajectory Truth", truth),
  ].join("");
  const claim = fuzz.oracle_status === "proved"
    ? "Sanitizer-backed crash observed. The dashboard reports evidence boundaries, not disclosure claims."
    : "No zero-day claim. Proof requires sanitizer-backed crash evidence from a loaded hunt report.";
  setText("[data-role='claim-copy']", claim);
}
function renderInstances(report, trajectory) {
  const candidates = trajectory?.candidates || [];
  const scores = Object.fromEntries((report.candidate_scores || []).map((score) => [score.candidate_id, score]));
  $("[data-role='instance-grid']").innerHTML = candidates.length ? candidates.map((candidate) => {
    const score = scores[candidate.candidate_id] || {};
    const selected = candidate.candidate_id === report.selected_candidate_id;
    return `<article class="instance-card ${selected ? "selected" : ""}">
      <div class="card-head"><h3>${escapeHtml(candidate.candidate_id)}</h3><span class="badge">${selected ? "selected" : trajectoryMode(candidate)}</span></div>
      <dl class="mini-metrics">
        ${metric("Score", number(score.score))}
        ${metric("API", percent(score.api_coverage))}
        ${metric("NFE", candidate.nfe)}
        ${metric("Steps", candidate.steps?.length || 0)}
      </dl>
      <details>
        <summary>Trajectory, source, and hash evidence</summary>
        ${trajectoryTable(candidate)}
        <pre class="source-view"><code>${escapeHtml(candidate.source)}</code></pre>
        <table><tbody>${hashRow("source_sha256", candidate.source_sha256)}</tbody></table>
      </details>
    </article>`;
  }).join("") : `<div class="empty-state">No candidate instances loaded from the matched bank.</div>`;
}
function renderCoverage(totals) {
  $("[data-role='coverage-grid']").innerHTML = [
    metric("Lines", percent(totals?.lines)),
    metric("Functions", percent(totals?.functions)),
    metric("Branches", percent(totals?.branches)),
    metric("Regions", percent(totals?.regions)),
  ].join("");
}
function renderFuzz(report) {
  const fuzz = report.fuzz || {};
  $("[data-role='fuzz-grid']").innerHTML = [
    metric("Build", fuzz.build_status || "n/a"),
    metric("Fuzz", fuzz.fuzz_status || "n/a"),
    metric("Oracle", fuzz.oracle_status || "n/a"),
    metric("Executed Units", value(fuzz.executed_units)),
    metric("Edges", value(fuzz.edge_coverage)),
    metric("Features", value(fuzz.feature_coverage)),
    metric("Exec/s", value(fuzz.executions_per_second)),
    metric("Sanitizer", fuzz.sanitizer || "n/a"),
  ].join("");
}
function renderEvidence(report) {
  const fuzz = report.fuzz || {};
  const rows = artifacts(report).map((artifact) => `<tr>
    <td>${escapeHtml(artifact.kind)}</td>
    <td>${escapeHtml(fuzz.oracle_status || "stored")}</td>
    <td>${escapeHtml(artifact.name)}</td>
    <td>${artifact.size}</td>
    <td><button class="copy-button" type="button" data-copy-hash="${artifact.sha256}" aria-live="polite">Copy hash</button></td>
  </tr>`);
  $("[data-role='evidence-body']").innerHTML = rows.length ? rows.join("") : `<tr><td colspan="5">No fuzz artifacts recorded.</td></tr>`;
}
function renderEvidenceDetails(report, trajectory) {
  const selected = selectedCandidate(report, trajectory);
  $("[data-role='trajectory-grid']").innerHTML = selected ? trajectoryTable(selected) : `<div class="empty-state">No selected candidate trajectory loaded.</div>`;
  setText("[data-role='source-code']", selected?.source || "No selected source loaded.");
  $("[data-role='hash-body']").innerHTML = hashRows(report, selected);
}
function trajectoryTable(candidate) {
  const rows = (candidate.steps || []).map((step) => `<tr><td>${step.step_index}</td><td>${step.masked_tokens_before}</td><td>${step.committed_tokens}</td><td>${number(step.mean_confidence)}</td><td>${escapeHtml(step.state_sha256)}</td></tr>`);
  return `<div class="table-wrap"><table><caption>${trajectoryMode(candidate)} trajectory</caption><thead><tr><th>Step</th><th>Masks</th><th>Commits</th><th>Confidence</th><th>State hash</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}
function hashRows(report, selected) {
  const coverage = report.coverage || {}, fuzz = report.fuzz || {};
  return [
    hashRow("candidate_bank_sha256", report.candidate_bank_sha256),
    hashRow("selected_harness_sha256", report.selected_harness_sha256),
    hashRow("selected_source_sha256", selected?.source_sha256),
    hashRow("coverage_build_log_sha256", coverage.build_log_sha256),
    hashRow("coverage_run_log_sha256", coverage.run_log_sha256),
    hashRow("fuzz_log_sha256", fuzz.fuzz_log_sha256),
    ...artifacts(report).map((artifact) => hashRow(`${artifact.kind}:${artifact.name}`, artifact.sha256)),
  ].join("");
}
function hashRow(label, hash) {
  return `<tr><th>${escapeHtml(label)}</th><td>${hash ? escapeHtml(hash) : "n/a"}</td></tr>`;
}
function artifacts(report) {
  const fuzz = report?.fuzz || {};
  return [
    ...(fuzz.corpus_artifacts || []).map((item) => ({ ...item, kind: "corpus" })),
    ...(fuzz.crash_artifacts || []).map((item) => ({ ...item, kind: "crash" })),
  ];
}
async function copyHash(event) {
  const target = event.target instanceof Element ? event.target.closest("[data-copy-hash]") : null;
  if (!(target instanceof HTMLButtonElement)) return;
  try {
    await navigator.clipboard.writeText(target.dataset.copyHash || "");
    target.textContent = "Copied";
  } catch {
    target.textContent = "Copy unavailable";
  }
}
function selectedCandidate(report, trajectory) {
  return (trajectory?.candidates || []).find((candidate) => candidate.candidate_id === report?.selected_candidate_id);
}
function trajectoryMode(candidate) {
  return (candidate?.steps || []).some((step) => Array.isArray(step.transfers) && step.transfers.length) ? "exact-token" : "aggregate";
}
function artifactCount(report) {
  return artifacts(report).length || "0";
}
function targetName(report) {
  return report?.target?.repository_url?.replace(/^https?:\/\/github.com\//, "") || "n/a";
}
function statusClass(status) {
  if (status === "completed" || status === "succeeded") return "status-proved";
  if (status === "running") return "status-running";
  if (status === "bank mismatch") return "status-warning";
  return "";
}
function number(value) {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}
function value(item) { return item === null || item === undefined ? "n/a" : item; }
load();
