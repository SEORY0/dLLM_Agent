export async function loadReport(api, cache, resourceId) {
  if (!resourceId) return null;
  if (Object.hasOwn(cache, resourceId)) return cache[resourceId];
  try {
    cache[resourceId] = await api(`/reports/${encodeURIComponent(resourceId)}`);
  } catch {
    cache[resourceId] = null;
  }
  return cache[resourceId];
}

async function matchingBankId(catalog, reportId, api, cache) {
  const report = await loadReport(api, cache, reportId);
  if (!report) return "";
  return catalog.resources.find((resource) => (
    resource.kind === "candidate_bank" && resource.sha256 === report.candidate_bank_sha256
  ))?.resource_id || "";
}

async function matchingReportId(catalog, bankId, api, cache) {
  const bank = catalog.resources.find((resource) => resource.resource_id === bankId);
  if (!bank) return "";
  for (const resource of catalog.resources.filter((item) => item.kind === "hunt_report")) {
    const report = await loadReport(api, cache, resource.resource_id);
    if (report?.candidate_bank_sha256 === bank.sha256) return resource.resource_id;
  }
  return "";
}

export async function alignBankToSelectedReport(catalog, api, cache) {
  const reportId = $("[data-role='report-select']")?.value;
  const select = $("[data-role='bank-select']");
  if (!reportId || !select) return;
  const bankId = await matchingBankId(catalog, reportId, api, cache);
  if (bankId) select.value = bankId;
  else select.selectedIndex = -1;
}

export async function alignReportToSelectedBank(catalog, api, cache) {
  const bankId = $("[data-role='bank-select']")?.value;
  const select = $("[data-role='report-select']");
  if (!bankId || !select) return;
  const reportId = await matchingReportId(catalog, bankId, api, cache);
  if (reportId) select.value = reportId;
  else select.selectedIndex = -1;
}
import { $ } from "./ui.js";
