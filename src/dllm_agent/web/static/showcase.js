const $ = (s, r = document) => r.querySelector(s);
const esc = (x) => String(x).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
const bucket = (v) => Math.max(1, Math.min(5, Math.ceil((v || 0) * 5)));

function makeSteps(exact, count) {
  const words = ["ret", "bytes", "cursor", "next", "ec", "json", "size", "data"];
  return Array.from({ length: count }, (_, i) => ({
    step_index: i + 1,
    masked_tokens_before: 4 - (i % 4),
    committed_tokens: 1,
    mean_confidence: Math.min(0.999, 0.68 + i / count * 0.31),
    state_sha256: String(i + 1).padStart(64, "0"),
    ...(exact ? { transfers: [{ generated_position: i % 40, token_id: 1000 + i, text: words[i % words.length], confidence: Math.min(0.999, 0.72 + i / count * 0.27) }] } : {})
  }));
}

const demo = [
  { steps: makeSteps(true, 24) },
  { steps: makeSteps(false, 24) }
];

function renderExact(candidate, stepIndex) {
  const step = candidate.steps[stepIndex];
  const current = new Set(step.transfers.map((t) => t.generated_position));
  const seen = new Map();
  candidate.steps.slice(0, stepIndex + 1).flatMap((s) => s.transfers).forEach((t) => seen.set(t.generated_position, t));
  return Array.from({ length: 40 }, (_, i) => {
    const t = seen.get(i);
    return `<span class="token-cell ${t ? "committed" : "masked"} ${current.has(i) ? "new" : ""} conf-${bucket(t?.confidence || step.mean_confidence)}">${esc(t?.text || "mask")}</span>`;
  }).join("");
}

function renderAggregate(candidate, stepIndex) {
  const step = candidate.steps[stepIndex], committed = Math.min(30, stepIndex + step.committed_tokens);
  return Array.from({ length: 30 }, (_, i) => `<span class="token-cell ${i < committed ? "committed" : "masked"} ${i >= stepIndex && i < committed ? "new" : ""} conf-${bucket(step.mean_confidence)}">${i < committed ? `step-${step.step_index}` : "mask"}</span>`).join("");
}

$(".exact-demo").innerHTML = renderExact(demo[0], 7);
$(".aggregate-demo").innerHTML = renderAggregate(demo[1], 7);
