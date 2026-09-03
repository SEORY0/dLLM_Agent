export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export const api = (path, options) => fetch(`/api/v1${path}`, {
  headers: { "content-type": "application/json" },
  ...options
}).then((response) => response.ok ? response.json() : Promise.reject(response));

export function metric(key, value) {
  return `<div><dt>${key}</dt><dd>${value}</dd></div>`;
}

export function percent(value) {
  return typeof value === "number" ? `${(value * (value <= 1 ? 100 : 1)).toFixed(1)}%` : "n/a";
}

export function confidenceBucket(value) {
  return Math.max(1, Math.min(5, Math.ceil((value || 0) * 5)));
}

export function capitalize(value) {
  return value[0].toUpperCase() + value.slice(1);
}

export function stageState(stage, current, stages) {
  return stages.indexOf(stage) < stages.indexOf(current) ? "complete" : stage === current ? "active" : "waiting";
}

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (match) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[match]);
}

export function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}
