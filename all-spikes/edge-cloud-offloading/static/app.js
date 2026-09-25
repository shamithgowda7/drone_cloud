/* ================================================================
   DroneCloud — ProAdapt Frontend Application
   4-Strategy Ablation Dashboard with Leaflet + Chart.js
   ================================================================ */

// ── State ─────────────────────────────────────────────────────
let ws = null;
let map = null;
let droneMarkers = {};
let charts = {};
let totalEventsReceived = 0;

// chart history buffers
const histTicks = [];
const histProadaptTime = [];
const histReactiveTime = [];
const histHeuristicTime = [];
const histStaticTime = [];

const histProadaptHit = [];
const histReactiveHit = [];
const histHeuristicHit = [];
const histStaticHit = [];

const MAX_HISTORY = 50;

// ── Leaflet Map ───────────────────────────────────────────────
function initMap() {
  map = L.map('map', {
    center: [12.9716, 77.5946],
    zoom: 13,
    zoomControl: true,
    attributionControl: false,
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '',
  }).addTo(map);

  // City center marker
  L.circleMarker([12.9716, 77.5946], {
    radius: 6,
    color: '#06b6d4',
    fillColor: '#06b6d4',
    fillOpacity: 0.4,
    weight: 2,
  }).addTo(map).bindTooltip('Cloud Base Station', {
    permanent: false,
    className: 'base-tooltip',
  });

  // Coverage radius
  L.circle([12.9716, 77.5946], {
    radius: 5000,
    color: 'rgba(6,182,212,0.15)',
    fillColor: 'rgba(6,182,212,0.04)',
    weight: 1,
    dashArray: '6 4',
  }).addTo(map);
}

function droneIcon(status, battery) {
  let color = '#10b981'; // green
  if (status === 'charging') color = '#38bdf8';
  else if (status === 'low_battery') color = '#f59e0b';
  else if (battery < 30) color = '#f43f5e';

  return L.divIcon({
    className: 'drone-marker',
    html: `
      <div style="position:relative;width:28px;height:28px;">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none" style="filter: drop-shadow(0 0 6px ${color}80);">
          <polygon points="14,2 24,22 14,17 4,22" fill="${color}" opacity="0.85"/>
          <circle cx="14" cy="14" r="3" fill="${color}" />
        </svg>
        <div style="position:absolute;bottom:-12px;left:50%;transform:translateX(-50%);
                    font-size:9px;font-family:'JetBrains Mono',monospace;color:${color};
                    white-space:nowrap;font-weight:600;
                    text-shadow: 0 0 4px rgba(0,0,0,0.8);">
          ${Math.round(battery)}%
        </div>
      </div>
    `,
    iconSize: [28, 40],
    iconAnchor: [14, 14],
  });
}

function updateDrones(drones) {
  const active = drones.filter(d => d.status !== 'charging').length;
  document.getElementById('activeDrones').textContent = `${active} active`;

  drones.forEach(d => {
    const icon = droneIcon(d.status, d.battery);

    if (droneMarkers[d.id]) {
      droneMarkers[d.id].setLatLng([d.lat, d.lng]);
      droneMarkers[d.id].setIcon(icon);
    } else {
      const marker = L.marker([d.lat, d.lng], { icon })
        .addTo(map)
        .bindTooltip('', { permanent: false, className: 'drone-tip' });
      droneMarkers[d.id] = marker;
    }

    droneMarkers[d.id].setTooltipContent(
      `<strong>${d.id}</strong><br>` +
      `Status: ${d.status}<br>` +
      `Battery: ${d.battery}%<br>` +
      `Altitude: ${d.altitude}m<br>` +
      `Edge Cap: ${d.edge_cap}`
    );
  });
}

// ── Charts ────────────────────────────────────────────────────
const CHART_COLORS = {
  cyan:    '#06b6d4',
  violet:  '#8b5cf6',
  emerald: '#10b981',
  amber:   '#f59e0b',
  rose:    '#f43f5e',
  sky:     '#38bdf8',
};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(100,140,200,0.08)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;

function initCharts() {
  // 1. Avg Completion Time (4-strategy)
  charts.time = new Chart(document.getElementById('chartTime'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'ProAdapt (Ours)',
          data: [],
          borderColor: CHART_COLORS.cyan,
          backgroundColor: 'rgba(6,182,212,0.12)',
          borderWidth: 2.5,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Reactive-Q',
          data: [],
          borderColor: CHART_COLORS.violet,
          borderWidth: 1.8,
          borderDash: [4, 4],
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Heuristic',
          data: [],
          borderColor: CHART_COLORS.emerald,
          borderWidth: 1.8,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Static (Cloud)',
          data: [],
          borderColor: CHART_COLORS.rose,
          borderWidth: 1.8,
          tension: 0.3,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      scales: {
        y: {
          title: { display: true, text: 'ms', color: '#64748b' },
          grid: { color: 'rgba(100,140,200,0.06)' },
        },
        x: { grid: { display: false } },
      },
      plugins: { legend: { position: 'top', align: 'end' } },
    },
  });

  // 2. Deadline Hit Rate (4-strategy)
  charts.deadline = new Chart(document.getElementById('chartDeadline'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'ProAdapt (Ours)',
          data: [],
          borderColor: CHART_COLORS.cyan,
          borderWidth: 2.5,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Reactive-Q',
          data: [],
          borderColor: CHART_COLORS.violet,
          borderWidth: 1.8,
          borderDash: [4, 4],
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Heuristic',
          data: [],
          borderColor: CHART_COLORS.emerald,
          borderWidth: 1.8,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Static (Cloud)',
          data: [],
          borderColor: CHART_COLORS.amber,
          borderWidth: 1.8,
          tension: 0.3,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      scales: {
        y: {
          min: 0, max: 100,
          title: { display: true, text: '%', color: '#64748b' },
          grid: { color: 'rgba(100,140,200,0.06)' },
        },
        x: { grid: { display: false } },
      },
      plugins: { legend: { position: 'top', align: 'end' } },
    },
  });

  // 3. ProAdapt Edge vs Cloud doughnut
  charts.split = new Chart(document.getElementById('chartSplit'), {
    type: 'doughnut',
    data: {
      labels: ['Edge Exec', 'Cloud Offload'],
      datasets: [{
        data: [50, 50],
        backgroundColor: [CHART_COLORS.violet, CHART_COLORS.cyan],
        borderColor: 'transparent',
        borderWidth: 0,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: { legend: { position: 'bottom' } },
    },
  });

  // 4. Cloud Scale gauge
  charts.scale = new Chart(document.getElementById('chartScale'), {
    type: 'doughnut',
    data: {
      labels: ['Active Scale', 'Remaining'],
      datasets: [{
        data: [1, 2],
        backgroundColor: [CHART_COLORS.sky, 'rgba(100,140,200,0.08)'],
        borderColor: 'transparent',
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      circumference: 270,
      rotation: -135,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });
}

function pushHistorySnapshot(snap) {
  const label = `${snap.t}`;
  histTicks.push(label);

  const pa = snap.proadapt || {};
  const rq = snap.reactive_q || {};
  const h  = snap.heuristic || {};
  const s  = snap.static || {};

  histProadaptTime.push(pa.avg_time_ms || 0);
  histReactiveTime.push(rq.avg_time_ms || 0);
  histHeuristicTime.push(h.avg_time_ms || 0);
  histStaticTime.push(s.avg_time_ms || 0);

  histProadaptHit.push(pa.hit_rate || 0);
  histReactiveHit.push(rq.hit_rate || 0);
  histHeuristicHit.push(h.hit_rate || 0);
  histStaticHit.push(s.hit_rate || 0);

  if (histTicks.length > MAX_HISTORY) {
    histTicks.shift();
    histProadaptTime.shift();
    histReactiveTime.shift();
    histHeuristicTime.shift();
    histStaticTime.shift();

    histProadaptHit.shift();
    histReactiveHit.shift();
    histHeuristicHit.shift();
    histStaticHit.shift();
  }
}

function updateCharts(state) {
  if (state.history && state.history.length > 0) {
    const latest = state.history[state.history.length - 1];
    if (!histTicks.includes(`${latest.t}`)) {
      pushHistorySnapshot(latest);
    }
  }

  // Time chart
  charts.time.data.labels = [...histTicks];
  charts.time.data.datasets[0].data = [...histProadaptTime];
  charts.time.data.datasets[1].data = [...histReactiveTime];
  charts.time.data.datasets[2].data = [...histHeuristicTime];
  charts.time.data.datasets[3].data = [...histStaticTime];
  charts.time.update('none');

  // Deadline chart
  charts.deadline.data.labels = [...histTicks];
  charts.deadline.data.datasets[0].data = [...histProadaptHit];
  charts.deadline.data.datasets[1].data = [...histReactiveHit];
  charts.deadline.data.datasets[2].data = [...histHeuristicHit];
  charts.deadline.data.datasets[3].data = [...histStaticHit];
  charts.deadline.update('none');

  // Split doughnut
  const pa = state.strategies ? state.strategies.proadapt : state.adaptive;
  if (pa) {
    charts.split.data.datasets[0].data = [pa.edge || 1, pa.cloud || 1];
    charts.split.update('none');
  }

  // Scale gauge
  if (state.cloud) {
    const scaleVal = state.cloud.scale;
    charts.scale.data.datasets[0].data = [scaleVal, Math.max(0, 3 - scaleVal)];
    charts.scale.update('none');
  }
}

// ── Metrics & Table Updates ───────────────────────────────────
function updateMetricsAndML(state) {
  const pa = state.strategies ? state.strategies.proadapt : state.adaptive;
  const rq = state.strategies ? state.strategies.reactive_q : null;
  const h  = state.strategies ? state.strategies.heuristic : null;
  const st = state.strategies ? state.strategies.static : state.static;

  // Header and top cards
  document.getElementById('tickNum').textContent = state.tick;
  document.getElementById('vFleet').textContent = `${state.drones.length} UAVs`;
  document.getElementById('vBattery').textContent = `${state.fleet_battery}%`;
  document.getElementById('vTasks').textContent = pa ? pa.done : 0;
  document.getElementById('vDeadline').textContent = pa ? `${pa.hit_rate}%` : '—';
  document.getElementById('vCloud').textContent = `${Math.round(state.cloud.load * 100)}%`;
  document.getElementById('vScale').textContent = `${state.cloud.scale}×`;

  // Colour battery
  const batt = state.fleet_battery;
  const battEl = document.getElementById('vBattery');
  if (batt < 30) battEl.style.color = '#f43f5e';
  else if (batt < 50) battEl.style.color = '#f59e0b';
  else battEl.style.color = '#10b981';

  // ML Telemetry banner
  const ml = state.proadapt_stats;
  if (ml) {
    document.getElementById('qActive').textContent = ml.qtable_entries_active;
    document.getElementById('qEpsilon').textContent = ml.epsilon.toFixed(3);
    document.getElementById('qShifts').textContent = ml.proactive_shifts;
    document.getElementById('qOverhead').textContent = ml.avg_decision_time_us > 0 ? ml.avg_decision_time_us.toFixed(1) : '< 5.0';
  }

  // 4-Strategy Ablation Table
  if (pa) {
    document.getElementById('paHit').textContent = `${pa.hit_rate}%`;
    document.getElementById('paTime').textContent = `${pa.avg_time_ms} ms`;
    document.getElementById('paLat').textContent = `${pa.avg_lat_ms} ms`;
    document.getElementById('paEdge').textContent = `${pa.edge_pct}%`;
  }
  if (rq) {
    document.getElementById('rqHit').textContent = `${rq.hit_rate}%`;
    document.getElementById('rqTime').textContent = `${rq.avg_time_ms} ms`;
    document.getElementById('rqLat').textContent = `${rq.avg_lat_ms} ms`;
    document.getElementById('rqEdge').textContent = `${rq.edge_pct}%`;
  }
  if (h) {
    document.getElementById('hHit').textContent = `${h.hit_rate}%`;
    document.getElementById('hTime').textContent = `${h.avg_time_ms} ms`;
    document.getElementById('hLat').textContent = `${h.avg_lat_ms} ms`;
    document.getElementById('hEdge').textContent = `${h.edge_pct}%`;
  }
  if (st) {
    document.getElementById('sHit').textContent = `${st.hit_rate}%`;
    document.getElementById('sTime').textContent = `${st.avg_time_ms} ms`;
    document.getElementById('sLat').textContent = `${st.avg_lat_ms} ms`;
    document.getElementById('sEdge').textContent = `${st.edge_pct}%`;
  }
}

// ── Event Log ─────────────────────────────────────────────────
function updateEvents(events) {
  if (!events || events.length === 0) return;

  const log = document.getElementById('eventLog');
  totalEventsReceived += events.length;
  document.getElementById('eventCount').textContent = `${totalEventsReceived} events`;

  events.forEach(e => {
    const row = document.createElement('div');
    row.className = 'event-row';
    const shiftBadge = e.proactive_shift ? `<span class="ev-proactive">⚡ Shift</span>` : '';
    row.innerHTML = `
      <span class="ev-task">${e.task}</span>
      <span class="ev-type">${e.type.replace(/_/g, ' ')}${shiftBadge}</span>
      <span class="ev-drone">${e.drone}</span>
      <span class="ev-loc ${e.loc}">${e.loc}</span>
      <span class="ev-time">${e.time_ms}ms</span>
      <span class="ev-time">${e.deadline_ms}ms</span>
      <span class="ev-met ${e.met ? 'yes' : 'no'}">${e.met ? '✓' : '✗'}</span>
    `;
    log.prepend(row);
  });

  while (log.children.length > 50) {
    log.removeChild(log.lastChild);
  }
}

// ── WebSocket ─────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    const pill = document.getElementById('statusPill');
    pill.classList.add('live');
    pill.querySelector('span:last-child').textContent = 'Live (4-Strat)';
  };

  ws.onmessage = (msg) => {
    const state = JSON.parse(msg.data);
    updateDrones(state.drones);
    updateMetricsAndML(state);
    updateCharts(state);
    updateEvents(state.events);
  };

  ws.onclose = () => {
    const pill = document.getElementById('statusPill');
    pill.classList.remove('live');
    pill.querySelector('span:last-child').textContent = 'Disconnected';
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => ws.close();
}

// ── Boot ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initCharts();
  connectWS();
});
