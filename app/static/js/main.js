/* main.js — shared utilities */

function fmt_currency(val) {
  if (val >= 1e6) return '£' + (val/1e6).toFixed(2) + 'M';
  if (val >= 1e3) return '£' + (val/1e3).toFixed(1) + 'k';
  return '£' + parseFloat(val).toLocaleString('en-GB', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

function fmt_pct(val) { return parseFloat(val).toFixed(1) + '%'; }
function fmt_num(val) { return parseInt(val).toLocaleString(); }

const SEGMENT_COLORS = {
  'High-Value': { bg: '#eef2ff', color: '#3730a3', pill: 'pill-blue', label: 'High-Value 💎' },
  'Loyal':      { bg: '#ecfeff', color: '#0e7490', pill: 'pill-teal', label: 'Loyal 🌟' },
  'At-Risk':    { bg: '#fff1f2', color: '#be123c', pill: 'pill-coral', label: 'At-Risk ⚠️' },
  'Low-Value':  { bg: '#f3f4f6', color: '#4b5563', pill: 'pill-gray', label: 'Low-Value' },
};

const RISK_STYLES = {
  'High':   { bg: '#fff1f2', color: '#be123c', pill: 'pill-coral' },
  'Medium': { bg: '#fffbeb', color: '#92400e', pill: 'pill-amber' },
  'Low':    { bg: '#ecfdf5', color: '#065f46', pill: 'pill-green' },
};

const CHART_COLORS = ['#4361EE','#06b6d4','#f59e0b','#f43f5e','#7c3aed','#10b981','#ef4444','#8b5cf6','#14b8a6','#f97316'];

function makeChart(id, config) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  if (typeof Chart === 'undefined') { console.error('Chart.js not loaded'); return null; }
  Chart.defaults.font.family = "'DM Sans', sans-serif";
  Chart.defaults.font.size   = 12;
  Chart.defaults.color       = '#6b7280';
  return new Chart(ctx, config);
}
