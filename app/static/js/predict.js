/* predict.js — single & batch prediction logic */

// ── Slider sync ───────────────────────────────────────
function syncInput(name) {
  const slider = document.getElementById(name + '-slider');
  const input  = document.getElementById(name + '-input');
  input.value  = slider.value;
}

function syncSlider(name) {
  const slider = document.getElementById(name + '-slider');
  const input  = document.getElementById(name + '-input');
  const v = parseFloat(input.value);
  if (!isNaN(v)) slider.value = Math.min(v, slider.max);
}

// ── Single Prediction ─────────────────────────────────
async function runPredict() {
  const recency   = parseFloat(document.getElementById('recency-input').value);
  const frequency = parseFloat(document.getElementById('frequency-input').value);
  const monetary  = parseFloat(document.getElementById('monetary-input').value);

  if (isNaN(recency) || isNaN(frequency) || isNaN(monetary)) {
    alert('Please enter valid numbers for all fields.'); return;
  }

  const btn     = document.getElementById('predict-btn-text');
  const spinner = document.getElementById('predict-spinner');
  btn.textContent = 'Predicting…';
  spinner.classList.remove('hidden');

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recency, frequency, monetary }),
    });
    const data = await res.json();
    if (data.success) renderResult(data.result);
    else alert('Prediction error: ' + data.error);
  } catch (e) {
    alert('Network error: ' + e.message);
  } finally {
    btn.textContent = 'Run Prediction';
    spinner.classList.add('hidden');
  }
}

function renderResult(r) {
  const panel = document.getElementById('result-panel');
  panel.classList.remove('hidden');

  // Segment badge
  const segBadge = document.getElementById('result-segment-badge');
  const segStyle = SEGMENT_COLORS[r.segment] || { bg: '#f3f4f6', color: '#374151', label: r.segment };
  segBadge.textContent = segStyle.label || r.segment;
  segBadge.style.background = segStyle.bg;
  segBadge.style.color = segStyle.color;

  // Churn
  const pct = (r.churn_probability * 100).toFixed(1);
  document.getElementById('result-churn').textContent = pct + '%';
  const bar = document.getElementById('result-churn-bar');
  bar.style.width = pct + '%';
  const riskStyle = RISK_STYLES[r.churn_risk] || RISK_STYLES['Low'];
  bar.style.background = riskStyle.color;

  const riskTag = document.getElementById('result-risk-tag');
  riskTag.textContent = r.churn_risk + ' Risk';
  riskTag.style.background = riskStyle.bg;
  riskTag.style.color = riskStyle.color;

  // CLV
  document.getElementById('result-clv').textContent = fmt_currency(r.predicted_clv_gbp);

  // Insight
  const insight = buildInsight(r);
  document.getElementById('result-insight').innerHTML = insight;
}

function buildInsight(r) {
  const segments = {
    'High-Value': '🎯 This is a <strong>top-tier customer</strong>. Prioritise retention, VIP perks, and early product access.',
    'Loyal':      '🌱 A <strong>regular buyer</strong> with growth potential. Target with upsell campaigns and loyalty rewards.',
    'At-Risk':    '⚠️ This customer <strong>may be drifting away</strong>. Send a personalised re-engagement offer now.',
    'Low-Value':  '📉 <strong>Low engagement</strong>. Consider a low-cost win-back email or discount nudge.',
  };
  return segments[r.segment] || 'No insight available for this segment.';
}

// ── Batch Prediction ──────────────────────────────────
let batchResults = null;

function handleFile(file) {
  if (!file) return;
  processBatchFile(file);
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file && file.name.endsWith('.csv')) processBatchFile(file);
  else showBatchStatus('error', 'Please upload a .csv file.');
}

async function processBatchFile(file) {
  showBatchStatus('loading', `Processing ${file.name}…`);
  document.getElementById('batch-result').classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch('/api/batch', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.success) {
      batchResults = data.data;
      showBatchStatus('success', `✔ Scored ${fmt_num(data.count)} customers successfully`);
      renderBatchTable(data.data);
    } else {
      showBatchStatus('error', 'Error: ' + data.error);
    }
  } catch (e) {
    showBatchStatus('error', 'Network error: ' + e.message);
  }
}

function showBatchStatus(type, msg) {
  const el = document.getElementById('batch-status');
  el.className = 'batch-status ' + type;
  el.textContent = msg;
  el.classList.remove('hidden');
}

function renderBatchTable(data) {
  if (!data.length) return;
  const result = document.getElementById('batch-result');
  result.classList.remove('hidden');

  document.getElementById('batch-summary').textContent =
    `Showing first ${Math.min(data.length, 50)} of ${fmt_num(data.length)} results`;

  const cols = Object.keys(data[0]);
  const thead = document.getElementById('batch-thead');
  const tbody = document.getElementById('batch-tbody');

  thead.innerHTML = '<tr>' + cols.map(c =>
    `<th>${c}</th>`
  ).join('') + '</tr>';

  tbody.innerHTML = data.slice(0, 50).map(row => {
    return '<tr>' + cols.map(c => {
      let val = row[c];
      if (c === 'Segment') {
        const s = SEGMENT_COLORS[val] || {};
        return `<td><span class="pill ${s.pill || 'pill-gray'}">${val}</span></td>`;
      }
      if (c === 'ChurnRisk') {
        const s = RISK_STYLES[val] || {};
        return `<td><span class="pill ${s.pill || 'pill-gray'}">${val}</span></td>`;
      }
      if (c === 'ChurnProbability') return `<td>${(val*100).toFixed(1)}%</td>`;
      if (c === 'PredictedCLV_GBP' || c === 'Monetary') return `<td>${fmt_currency(val)}</td>`;
      return `<td>${val !== null && val !== undefined ? val : '—'}</td>`;
    }).join('') + '</tr>';
  }).join('');
}

function downloadResults() {
  if (!batchResults) return;
  const cols  = Object.keys(batchResults[0]);
  const csv   = [cols.join(','),
    ...batchResults.map(row => cols.map(c => `"${row[c]}"`).join(','))
  ].join('\n');
  const blob  = new Blob([csv], { type: 'text/csv' });
  const url   = URL.createObjectURL(blob);
  const a     = document.createElement('a');
  a.href = url; a.download = 'customer_predictions.csv'; a.click();
  URL.revokeObjectURL(url);
}
