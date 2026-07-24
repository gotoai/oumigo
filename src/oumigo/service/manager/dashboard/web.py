"""The dashboard's single HTML page (V1.0).

Mobile-first, theme-aware, self-contained except for a pinned Chart.js. The page
polls ``/api/sheets/gpu_util`` and draws one line per node. Colors come from the
data-viz reference categorical palette (fixed slot order, never cycled for the
first 8; a larger fleet is a known V1.0 limitation flagged in the JS).
"""

from __future__ import annotations

CHARTJS_SRC = "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>oumigo — fleet dashboard</title>
<script src="__CHARTJS_SRC__"></script>
<style>
  :root {
    --page: #f9f9f7; --surface: #fcfcfb;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      --page: #0d0d0d; --surface: #1a1a19;
      --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] {
    --page: #0d0d0d; --surface: #1a1a19;
    --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-text-size-adjust: 100%;
    padding: max(12px, env(safe-area-inset-top)) 12px 24px;
  }
  header { max-width: 960px; margin: 0 auto 12px; }
  h1 { font-size: 1.15rem; font-weight: 650; margin: 0 0 2px; }
  .sub { color: var(--text-secondary); font-size: 0.82rem; margin: 0; }
  .status { color: var(--muted); font-size: 0.75rem; margin-top: 4px;
            font-variant-numeric: tabular-nums; }
  .sheet {
    max-width: 960px; margin: 0 auto;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 12px 10px;
  }
  .sheet h2 { font-size: 0.95rem; font-weight: 600; margin: 0 0 2px; }
  .sheet .note { color: var(--text-secondary); font-size: 0.78rem; margin: 0 0 10px; }
  .chart-wrap { position: relative; height: min(62vh, 460px); min-height: 300px; }
  .empty {
    position: absolute; inset: 0; display: none;
    align-items: center; justify-content: center;
    color: var(--muted); font-size: 0.85rem; text-align: center; padding: 0 20px;
  }
</style>
</head>
<body>
<header>
  <h1>oumigo fleet dashboard</h1>
  <p class="sub">Cluster performance &middot; reporting plane (V1.0)</p>
  <p class="status" id="status">connecting&hellip;</p>
</header>

<section class="sheet">
  <h2>GPU utilization by node</h2>
  <p class="note">Average <code>gpu:#N_util_pct</code> per node &middot; last 60 min, 5s grid &middot; 6-sample moving average</p>
  <div class="chart-wrap">
    <canvas id="gpuUtil"></canvas>
    <div class="empty" id="empty">No GPU utilization reported yet.<br>Waiting for worker metrics&hellip;</div>
  </div>
</section>

<script>
(function () {
  "use strict";
  // Data-viz reference categorical palette (fixed slot order). Slots 1..8; a fleet
  // larger than 8 nodes cycles here (a known V1.0 limitation — the proper fix is
  // fold-to-"Other"/facet, deferred).
  var PALETTE = {
    light: ["#2a78d6","#008300","#e87ba4","#eda100","#1baf7a","#eb6834","#4a3aa7","#e34948"],
    dark:  ["#3987e5","#008300","#d55181","#c98500","#199e70","#d95926","#9085e9","#e66767"]
  };
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark") return true;
    if (t === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function ink(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  var chart = null;

  function buildOptions() {
    var muted = ink("--muted"), grid = ink("--grid"), text = ink("--text-secondary");
    return {
      responsive: true, maintainAspectRatio: false,
      animation: false, spanGaps: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { min: 0, max: 100, border: { display: false },
             ticks: { color: muted, callback: function (v) { return v + "%"; } },
             grid: { color: grid } },
        x: { border: { color: ink("--baseline") },
             ticks: { color: muted, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
             grid: { display: false } }
      },
      plugins: {
        legend: { labels: { color: text, boxWidth: 12, boxHeight: 2, usePointStyle: false } },
        tooltip: { callbacks: { label: function (c) {
          return c.dataset.label + ": " + (c.parsed.y == null ? "—" : c.parsed.y + "%");
        } } }
      }
    };
  }

  function toDatasets(series) {
    var pal = PALETTE[isDark() ? "dark" : "light"];
    return series.map(function (s, i) {
      var color = pal[i % pal.length];
      return {
        label: s.label, data: s.data,
        borderColor: color, backgroundColor: color,
        borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
        tension: 0.25, spanGaps: false
      };
    });
  }

  function render(payload) {
    var empty = document.getElementById("empty");
    empty.style.display = payload.series.length ? "none" : "flex";
    var datasets = toDatasets(payload.series);
    if (!chart) {
      chart = new Chart(document.getElementById("gpuUtil").getContext("2d"), {
        type: "line",
        data: { labels: payload.labels, datasets: datasets },
        options: buildOptions()
      });
    } else {
      chart.data.labels = payload.labels;
      chart.data.datasets = datasets;
      chart.options = buildOptions();
      chart.update();
    }
  }

  function setStatus(ok, payload) {
    var el = document.getElementById("status");
    if (ok) {
      var n = payload.series.length;
      el.textContent = "updated " + payload.generated_at + " UTC · " +
                       n + " node" + (n === 1 ? "" : "s");
    } else {
      el.textContent = "reconnecting… (last update failed)";
    }
  }

  function poll() {
    fetch("/api/sheets/gpu_util", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (p) { render(p); setStatus(true, p); })
      .catch(function () { setStatus(false, null); });
  }

  poll();
  setInterval(poll, 5000);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", poll);
  }
})();
</script>
</body>
</html>
""".replace("__CHARTJS_SRC__", CHARTJS_SRC)
