/**
 * Predictive Analytics & Future Trend Forecasting Logic
 * Connects historical time-series data with ML predictions, confidence intervals, and interactive charts
 */

let forecastPayload = null;
let currentHorizon = 60;
let showConfidence = true;
let currentTablePage = 1;
const pageSize = 10;
let searchQuery = '';

// Chart instances
let mainForecastChart = null;
let featureChart = null;
let smoothingChartInstance = null;

document.addEventListener('DOMContentLoaded', async () => {
  setupListeners();
  await loadForecastData();
});

async function loadForecastData() {
  try {
    const res = await fetch('forecast_data.json');
    forecastPayload = await res.json();

    updateScorecard(forecastPayload.modelMetrics);
    renderAllVisualizations();
    updateKPIs();
    renderTable();
    showToast(`Loaded ${forecastPayload.historical.length} historical actuals & 60-day ML predictions`);
  } catch (err) {
    console.error('Error loading forecast data:', err);
    showToast('Failed to load forecast data', true);
  }
}

function updateScorecard(metrics) {
  document.getElementById('val-accuracy').textContent = `${metrics.AccuracyPct}%`;
  document.getElementById('val-mape').textContent = `${metrics.MAPE}%`;
  document.getElementById('val-r2').textContent = metrics.R2_Score;
  document.getElementById('val-mae').textContent = '$' + metrics.MAE.toLocaleString();
  document.getElementById('val-rmse').textContent = '$' + metrics.RMSE.toLocaleString();
  document.getElementById('kpi-stat-mae').textContent = '$' + metrics.MAE.toLocaleString();
}

function updateKPIs() {
  if (!forecastPayload) return;
  const slicedForecast = forecastPayload.forecast.slice(0, currentHorizon);
  const totalProjected = slicedForecast.reduce((acc, f) => acc + f.ForecastRevenue, 0);
  const avgDaily = slicedForecast.length ? Math.round(totalProjected / slicedForecast.length) : 0;
  const totalUnits = slicedForecast.reduce((acc, f) => acc + f.EstimatedUnits, 0);
  const avgUnits = slicedForecast.length ? Math.round(totalUnits / slicedForecast.length) : 0;

  document.getElementById('kpi-projected-rev').textContent = 
    '$' + (totalProjected >= 1000000 ? (totalProjected/1000000).toFixed(2) + 'M' : (totalProjected/1000).toFixed(0) + 'k');
  document.getElementById('kpi-avg-daily-rev').textContent = `$${avgDaily.toLocaleString()} / day`;
  document.getElementById('kpi-demand-units').textContent = `~${avgUnits.toLocaleString()} units / day`;
}

function renderAllVisualizations() {
  renderMainForecastChart();
  renderFeatureImportance();
  renderSmoothingChart();
}

// Chart 1: Main Forecast with Confidence Bounds
function renderMainForecastChart() {
  const ctx = document.getElementById('forecastMainChart').getContext('2d');
  const historical = forecastPayload.historical;
  const forecast = forecastPayload.forecast.slice(0, currentHorizon);

  const histLabels = historical.map(h => h.Date);
  const foreLabels = forecast.map(f => f.Date);
  const allLabels = [...histLabels, ...foreLabels];

  // Historical data points
  const histValues = [...historical.map(h => h.Revenue), ...new Array(foreLabels.length).fill(null)];

  // Forecast data points (connect with last historical point for continuous curve)
  const lastHistRev = historical[historical.length - 1].Revenue;
  const foreValues = [
    ...new Array(histLabels.length - 1).fill(null),
    lastHistRev,
    ...forecast.map(f => f.ForecastRevenue)
  ];

  // Upper and Lower confidence bounds
  const upperBounds = [
    ...new Array(histLabels.length - 1).fill(null),
    lastHistRev,
    ...forecast.map(f => f.UpperBound)
  ];
  const lowerBounds = [
    ...new Array(histLabels.length - 1).fill(null),
    lastHistRev,
    ...forecast.map(f => f.LowerBound)
  ];

  if (mainForecastChart) mainForecastChart.destroy();

  const isDark = document.body.classList.contains('dark-theme');
  const textColor = isDark ? '#94a3b8' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

  const datasets = [
    {
      label: 'Historical Actuals',
      data: histValues,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.08)',
      fill: false,
      tension: 0.25,
      borderWidth: 2,
      pointRadius: 1,
      pointHoverRadius: 4
    },
    {
      label: 'ML Forecast',
      data: foreValues,
      borderColor: '#8b5cf6',
      backgroundColor: 'rgba(139, 92, 246, 0.15)',
      fill: false,
      tension: 0.3,
      borderWidth: 2.5,
      pointRadius: 2,
      pointHoverRadius: 5
    }
  ];

  if (showConfidence) {
    datasets.push(
      {
        label: 'Upper Bound (95% CI)',
        data: upperBounds,
        borderColor: 'rgba(139, 92, 246, 0.3)',
        borderDash: [4, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: '+1',
        backgroundColor: 'rgba(139, 92, 246, 0.1)'
      },
      {
        label: 'Lower Bound (95% CI)',
        data: lowerBounds,
        borderColor: 'rgba(139, 92, 246, 0.3)',
        borderDash: [4, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      }
    );
  }

  mainForecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: textColor, font: { family: 'Plus Jakarta Sans', size: 11 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.raw === null) return '';
              return ` ${ctx.dataset.label}: $${Number(ctx.raw).toLocaleString()}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, maxTicksLimit: 12 }
        },
        y: {
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            callback: (v) => '$' + (v / 1000).toFixed(0) + 'k'
          }
        }
      }
    }
  });
}

// Chart 2: Feature Drivers (Feature Importance)
function renderFeatureImportance() {
  const ctx = document.getElementById('featureImportanceChart').getContext('2d');
  const featData = forecastPayload.featureImportance;

  if (featureChart) featureChart.destroy();
  const isDark = document.body.classList.contains('dark-theme');
  const textColor = isDark ? '#94a3b8' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

  featureChart = new Chart(ctx, {
    type: 'bar',
    indexAxis: 'y',
    data: {
      labels: featData.map(f => f.feature),
      datasets: [{
        label: 'Impact Weight %',
        data: featData.map(f => f.importance),
        backgroundColor: ['#8b5cf6', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b'],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: (ctx) => ` Relative Impact: ${ctx.raw}%` }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, callback: v => v + '%' }
        },
        y: {
          grid: { display: false },
          ticks: { color: textColor, font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      }
    }
  });
}

// Chart 3: Historical Trend Smoothing (Raw vs 7d vs 30d MA)
function renderSmoothingChart() {
  const ctx = document.getElementById('smoothingChart').getContext('2d');
  const hist = forecastPayload.historical.slice(-60);

  if (smoothingChartInstance) smoothingChartInstance.destroy();
  const isDark = document.body.classList.contains('dark-theme');
  const textColor = isDark ? '#94a3b8' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

  smoothingChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: hist.map(h => h.Date),
      datasets: [
        {
          label: 'Raw Revenue',
          data: hist.map(h => h.Revenue),
          borderColor: 'rgba(148, 163, 184, 0.4)',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: '7-Day Moving Avg',
          data: hist.map(h => h.Revenue_MA7),
          borderColor: '#06b6d4',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3
        },
        {
          label: '30-Day Moving Avg',
          data: hist.map(h => h.Revenue_MA30),
          borderColor: '#f59e0b',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: textColor, font: { family: 'Plus Jakarta Sans', size: 10 } }
        }
      },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: textColor, maxTicksLimit: 6 } },
        y: { grid: { color: gridColor }, ticks: { color: textColor, callback: v => '$' + (v/1000).toFixed(0) + 'k' } }
      }
    }
  });
}

// Table Rendering
function renderTable() {
  if (!forecastPayload) return;
  const forecast = forecastPayload.forecast.slice(0, currentHorizon);
  const filtered = forecast.filter(f => {
    if (!searchQuery) return true;
    return f.Date.includes(searchQuery) || f.DayOfWeek.toLowerCase().includes(searchQuery);
  });

  const tbody = document.getElementById('forecast-table-body');
  tbody.innerHTML = '';

  const totalPages = Math.ceil(filtered.length / pageSize) || 1;
  if (currentTablePage > totalPages) currentTablePage = totalPages;

  const startIdx = (currentTablePage - 1) * pageSize;
  const pageItems = filtered.slice(startIdx, startIdx + pageSize);

  pageItems.forEach(item => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${item.Date}</strong></td>
      <td>${item.DayOfWeek}</td>
      <td class="right font-medium text-purple">$${item.ForecastRevenue.toLocaleString()}</td>
      <td class="right text-dim">$${item.LowerBound.toLocaleString()}</td>
      <td class="right text-dim">$${item.UpperBound.toLocaleString()}</td>
      <td class="right font-medium">${item.EstimatedUnits.toLocaleString()} units</td>
      <td><span class="status-chip"><i class="fa-solid fa-circle-check"></i> High (95% CI)</span></td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('table-record-info').textContent = 
    `Showing ${filtered.length > 0 ? startIdx + 1 : 0} - ${Math.min(startIdx + pageSize, filtered.length)} of ${filtered.length} forecasted days`;
  document.getElementById('p-text').textContent = `Page ${currentTablePage} of ${totalPages}`;
  document.getElementById('p-prev').disabled = currentTablePage === 1;
  document.getElementById('p-next').disabled = currentTablePage >= totalPages;
}

function setupListeners() {
  // Horizon Select
  document.getElementById('horizon-select').addEventListener('change', (e) => {
    currentHorizon = parseInt(e.target.value, 10);
    renderMainForecastChart();
    updateKPIs();
    renderTable();
    showToast(`Horizon updated to next ${currentHorizon} days`);
  });

  // Confidence Interval Toggle
  document.getElementById('confidence-toggle').addEventListener('change', (e) => {
    showConfidence = e.target.value === 'true';
    renderMainForecastChart();
  });

  // Search filter
  document.getElementById('forecast-search').addEventListener('input', (e) => {
    searchQuery = e.target.value.toLowerCase().trim();
    currentTablePage = 1;
    renderTable();
  });

  // Pagination
  document.getElementById('p-prev').addEventListener('click', () => {
    if (currentTablePage > 1) {
      currentTablePage--;
      renderTable();
    }
  });

  document.getElementById('p-next').addEventListener('click', () => {
    const filtered = forecastPayload.forecast.slice(0, currentHorizon);
    const totalPages = Math.ceil(filtered.length / pageSize);
    if (currentTablePage < totalPages) {
      currentTablePage++;
      renderTable();
    }
  });

  // Theme Toggle
  document.getElementById('theme-btn').addEventListener('click', () => {
    document.body.classList.toggle('light-theme');
    document.body.classList.toggle('dark-theme');
    const isDark = document.body.classList.contains('dark-theme');
    document.querySelector('#theme-btn i').className = isDark ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    renderAllVisualizations();
  });

  // Export to Excel
  document.getElementById('export-forecast-btn').addEventListener('click', () => {
    if (!forecastPayload) return;
    const ws = XLSX.utils.json_to_sheet(forecastPayload.forecast.slice(0, currentHorizon));
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'ML_Forecast_Data');
    XLSX.writeFile(wb, 'Predictive_Analytics_Forecast_Report.xlsx');
    showToast('Forecast data exported to Excel');
  });
}

function showToast(msg, isError = false) {
  const toast = document.getElementById('toast');
  const txt = document.getElementById('toast-text');
  txt.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}
