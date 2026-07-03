/* customers.js — paginated customer explorer */

let currentPage = 1;
let totalPages  = 1;
const PER_PAGE  = 20;

async function loadCustomers() {
  const segment = document.getElementById('filter-segment').value;
  const sort    = document.getElementById('filter-sort').value;
  const order   = document.getElementById('filter-order').value;

  const url = `/api/customers?page=${currentPage}&per_page=${PER_PAGE}&segment=${segment}&sort=${sort}&order=${order}`;

  document.getElementById('customers-tbody').innerHTML =
    '<tr><td colspan="7" class="loading-row">Loading…</td></tr>';

  const res  = await fetch(url);
  const data = await res.json();

  totalPages = Math.ceil(data.total / PER_PAGE);

  document.getElementById('header-count').textContent =
    `${fmt_num(data.total)} customers`;

  renderTable(data.data);
  renderPagination(data.total);
}

function renderTable(rows) {
  const tbody = document.getElementById('customers-tbody');

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No customers found.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const seg      = r.Segment || '—';
    const segStyle = SEGMENT_COLORS[seg] || { pill: 'pill-gray', label: seg };

    const churnProb = r.ChurnProbability !== undefined ? (r.ChurnProbability * 100).toFixed(1) + '%' : '—';
    const churnRisk = getChurnRisk(r.ChurnProbability);
    const riskStyle = RISK_STYLES[churnRisk] || {};

    const clv = r.PredictedCLV_GBP !== undefined ? fmt_currency(r.PredictedCLV_GBP) : '—';

    return `
      <tr>
        <td><strong>#${r.CustomerID}</strong></td>
        <td><span class="pill ${segStyle.pill || 'pill-gray'}">${segStyle.label || seg}</span></td>
        <td>${r.Recency} days</td>
        <td>${r.Frequency} orders</td>
        <td>${fmt_currency(r.Monetary)}</td>
        <td>
          <span class="pill ${riskStyle.pill || 'pill-gray'}">${churnRisk} (${churnProb})</span>
        </td>
        <td>${clv}</td>
      </tr>`;
  }).join('');
}

function getChurnRisk(prob) {
  if (prob === undefined || prob === null) return '—';
  if (prob >= 0.7) return 'High';
  if (prob >= 0.4) return 'Medium';
  return 'Low';
}

function renderPagination(total) {
  const start = (currentPage - 1) * PER_PAGE + 1;
  const end   = Math.min(currentPage * PER_PAGE, total);
  document.getElementById('page-info').textContent =
    `${fmt_num(start)}–${fmt_num(end)} of ${fmt_num(total)}`;
  document.getElementById('prev-btn').disabled = currentPage <= 1;
  document.getElementById('next-btn').disabled = currentPage >= totalPages;
}

function changePage(delta) {
  const next = currentPage + delta;
  if (next < 1 || next > totalPages) return;
  currentPage = next;
  loadCustomers();
  document.querySelector('.table-card').scrollIntoView({ behavior: 'smooth' });
}

function applyFilters() {
  currentPage = 1;
  loadCustomers();
}

// Init
loadCustomers();
