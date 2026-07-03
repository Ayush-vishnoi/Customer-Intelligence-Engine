/* dashboard.js — loads all analytics data and renders charts */

async function fetchAll() {
  const safe = url => fetch(url).then(r => r.json()).catch(() => null);
  const [kpis, segments, revenue, churn, clv, pareto, products] = await Promise.all([
    safe('/api/kpis'),
    safe('/api/segments'),
    safe('/api/revenue-monthly'),
    safe('/api/churn-stats'),
    safe('/api/clv-distribution'),
    safe('/api/pareto'),
    safe('/api/top-products'),
  ]);
  return { kpis, segments, revenue, churn, clv, pareto, products };
}

function renderKPIs(kpis) {
  document.getElementById('kpi-revenue-val').textContent   = fmt_currency(kpis.total_revenue);
  document.getElementById('kpi-revenue-sub').textContent   = 'across all customers';
  document.getElementById('kpi-customers-val').textContent = fmt_num(kpis.n_customers);
  document.getElementById('kpi-customers-sub').textContent = '37 countries';
  document.getElementById('kpi-churn-val').textContent     = fmt_pct(kpis.churn_rate);
  document.getElementById('kpi-clv-val').textContent       = fmt_currency(kpis.avg_clv);
  document.getElementById('kpi-pareto-val').textContent    = fmt_pct(kpis.top_pct_80);
  document.getElementById('kpi-loyalty-val').textContent   = kpis.loyalty_mult + '×';

  // Animate card entries
  document.querySelectorAll('.kpi-card').forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(12px)';
    setTimeout(() => {
      card.style.transition = 'opacity .3s ease, transform .3s ease';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, i * 60);
  });
}

function renderRevenue(revenue) {
  if (!revenue.labels.length) return;
  makeChart('revenueChart', {
    type: 'line',
    data: {
      labels: revenue.labels,
      datasets: [{
        label: 'Monthly Revenue',
        data: revenue.values,
        borderColor: '#4361EE',
        backgroundColor: 'rgba(67,97,238,.07)',
        borderWidth: 2.5,
        pointRadius: 3,
        pointBackgroundColor: '#4361EE',
        tension: 0.35,
        fill: true,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ' ' + fmt_currency(ctx.raw) }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 40, maxTicksLimit: 8 } },
        y: {
          grid: { color: '#f1f3f6' },
          ticks: { callback: v => fmt_currency(v) }
        }
      }
    }
  });
}

function renderSegmentDonut(segments) {
  const colors = segments.map(s => {
    const map = { 'High-Value':'#4361EE','Loyal':'#06b6d4','At-Risk':'#f43f5e','Low-Value':'#9ca3af' };
    return map[s.Segment] || '#888';
  });
  makeChart('segmentDonut', {
    type: 'doughnut',
    data: {
      labels: segments.map(s => s.Segment),
      datasets: [{
        data: segments.map(s => s.count),
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: '#fff',
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true } },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${fmt_num(ctx.raw)} customers`
          }
        }
      }
    }
  });
}

function renderChurn(churn) {
  const colors = ['#10b981','#f59e0b','#f43f5e'];
  makeChart('churnChart', {
    type: 'bar',
    data: {
      labels: churn.labels,
      datasets: [{
        data: churn.counts,
        backgroundColor: colors,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + fmt_num(ctx.raw) + ' customers' } }
      },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f3f6' }, ticks: { precision: 0 } }
      }
    }
  });
}

function renderCLV(clv) {
  if (!clv.bins.length) return;
  makeChart('clvChart', {
    type: 'bar',
    data: {
      labels: clv.bins.map(b => fmt_currency(b)),
      datasets: [{
        data: clv.counts,
        backgroundColor: 'rgba(6,182,212,.6)',
        borderColor: '#06b6d4',
        borderWidth: 1,
        borderRadius: 3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => 'From ' + items[0].label,
            label: ctx => ' ' + fmt_num(ctx.raw) + ' customers',
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 30 } },
        y: { grid: { color: '#f1f3f6' } }
      }
    }
  });
}

function renderPareto(pareto) {
  makeChart('paretoChart', {
    type: 'line',
    data: {
      labels: pareto.cust_pct,
      datasets: [
        {
          label: 'Cumulative Revenue %',
          data: pareto.rev_pct,
          borderColor: '#4361EE',
          backgroundColor: 'rgba(67,97,238,.08)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
          fill: true,
        },
        {
          label: '80% Line',
          data: pareto.cust_pct.map(() => 80),
          borderColor: '#f43f5e',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => fmt_pct(items[0].label) + ' of customers',
            label: ctx => ' ' + fmt_pct(ctx.raw) + ' of revenue',
          }
        }
      },
      scales: {
        x: {
          grid: { color: '#f1f3f6' },
          ticks: { maxTicksLimit: 6, callback: v => fmt_pct(v) }
        },
        y: {
          grid: { color: '#f1f3f6' },
          ticks: { callback: v => fmt_pct(v) },
          min: 0, max: 100,
        }
      }
    }
  });
}

function renderProducts(products) {
  if (!products.length) return;
  makeChart('productsChart', {
    type: 'bar',
    data: {
      labels: products.map(p => p.product.length > 30 ? p.product.substring(0,28)+'…' : p.product),
      datasets: [{
        data: products.map(p => p.revenue),
        backgroundColor: CHART_COLORS.slice(0, products.length),
        borderRadius: 5,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + fmt_currency(ctx.raw) } }
      },
      scales: {
        x: { grid: { color: '#f1f3f6' }, ticks: { callback: v => fmt_currency(v) } },
        y: { grid: { display: false } }
      }
    }
  });
}

function renderSegmentTable(segments) {
  const container = document.getElementById('segment-table');
  const segStyles = {
    'High-Value': { emoji: '💎', pill: 'pill-blue' },
    'Loyal':      { emoji: '🌟', pill: 'pill-teal' },
    'At-Risk':    { emoji: '⚠️', pill: 'pill-coral' },
    'Low-Value':  { emoji: '',   pill: 'pill-gray' },
  };

  segments.forEach(s => {
    const style = segStyles[s.Segment] || { emoji: '', pill: 'pill-gray' };
    const row = document.createElement('div');
    row.className = 'seg-row';
    row.innerHTML = `
      <span><span class="pill ${style.pill}">${style.emoji} ${s.Segment}</span></span>
      <span>${fmt_num(s.count)}</span>
      <span>${s.revenue_pct}%</span>
      <span>${fmt_currency(s.avg_spend)}</span>
      <span>${fmt_pct(s.churn_rate * 100)}</span>
    `;
    container.appendChild(row);
  });
}

// ── Init ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetchAll().then(({ kpis, segments, revenue, churn, clv, pareto, products }) => {
    if (!kpis) {
      document.getElementById('kpi-grid').innerHTML = '<div style="color:#f43f5e;padding:20px;grid-column:1/-1">⚠️ Could not connect to server. Make sure Flask is running on port 8080.</div>';
      return;
    }
    renderKPIs(kpis);
    if (revenue)  renderRevenue(revenue);
    if (segments) { renderSegmentDonut(segments); renderSegmentTable(segments); }
    if (churn)    renderChurn(churn);
    if (clv)      renderCLV(clv);
    if (pareto)   renderPareto(pareto);
    if (products) renderProducts(products);
  }).catch(err => {
    console.error('Dashboard error:', err);
    document.getElementById('kpi-grid').innerHTML = '<div style="color:#f43f5e;padding:20px;grid-column:1/-1">⚠️ Dashboard error: ' + err.message + '</div>';
  });
});
