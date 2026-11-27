"""The console's pages: a two-pane shell (topics left, the topic's view right), the
sign-in and Create Admin Account pages, and the Workers and GPU Utilization views.

Self-contained except for a pinned Chart.js on the GPU page; theme-aware. Values are
escaped here; the views poll their JSON every 5 s.
"""

from __future__ import annotations

import json
from html import escape

from oumigo import __version__

CHARTJS_SRC = "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"

TOPICS = (
    ("workers", "/workers", "Workers"),
    ("gpu", "/gpu", "GPU Utilization"),
    ("tokens", "/tokens", "Token Generation"),
    ("tokens-cumulative", "/tokens/cumulative", "Token Generation (Cumulative)"),
)

_CSS = """
  :root {
    --page: #f9f9f7; --surface: #fcfcfb; --raised: #f1f0ec;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --accent: #2a78d6; --accent-text: #fff; --danger: #c2372f; --warn-bg: #fdf3dc;
    --warn-text: #6b4b00; --error-bg: #fde8e6; --error-text: #8a1f17;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      --page: #0d0d0d; --surface: #1a1a19; --raised: #242422;
      --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --accent: #3987e5; --danger: #e66767; --warn-bg: #3a2f12; --warn-text: #f3d48a;
      --error-bg: #3d1a17; --error-text: #f5b3ad;
    }
  }
  :root[data-theme="dark"] {
    --page: #0d0d0d; --surface: #1a1a19; --raised: #242422;
    --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --accent: #3987e5; --danger: #e66767; --warn-bg: #3a2f12; --warn-text: #f3d48a;
    --error-bg: #3d1a17; --error-text: #f5b3ad;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px;
  }
  a { color: var(--accent); }
  .top {
    display: flex; align-items: center; gap: 12px; padding: 10px 16px;
    border-bottom: 1px solid var(--border); background: var(--surface);
  }
  .top .brand { font-weight: 650; font-size: 1.05rem; }
  .top .right { margin-left: auto; display: flex; align-items: center; gap: 12px;
                color: var(--text-secondary); font-size: 0.85rem; }
  .top form { margin: 0; }
  .layout { display: flex; min-height: calc(100vh - 49px); }
  .side { width: 230px; flex: none; padding: 12px 8px; border-right: 1px solid var(--border); }
  .side a {
    display: block; padding: 8px 10px; border-radius: 8px; color: var(--text-primary);
    text-decoration: none; margin-bottom: 2px;
  }
  .side a:hover { background: var(--raised); }
  .side a.active { background: var(--raised); font-weight: 600; }
  main { flex: 1; min-width: 0; padding: 16px 20px 32px; }
  h1 { font-size: 1.1rem; font-weight: 650; margin: 0 0 4px; }
  .sub { color: var(--text-secondary); font-size: 0.82rem; margin: 0 0 12px; }
  .status { color: var(--muted); font-size: 0.78rem; font-variant-numeric: tabular-nums; }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px;
  }
  .notice { border-radius: 8px; padding: 10px 12px; margin: 0 0 12px;
            background: var(--warn-bg); color: var(--warn-text); }
  .error { border-radius: 8px; padding: 10px 12px; margin: 0 0 12px;
           background: var(--error-bg); color: var(--error-text); }
  .error ul { margin: 4px 0 0; padding-left: 18px; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid);
           white-space: nowrap; }
  th { font-size: 0.75rem; color: var(--text-secondary); font-weight: 600; }
  td.mono, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   font-size: 0.82rem; }
  button, .button {
    font: inherit; border-radius: 8px; padding: 6px 12px; cursor: pointer;
    border: 1px solid var(--border); background: var(--raised); color: var(--text-primary);
  }
  button.primary { background: var(--accent); color: var(--accent-text); border-color: transparent; }
  button.danger { color: var(--danger); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .muted { color: var(--muted); }
  form.stack { display: grid; gap: 10px; max-width: 360px; }
  form.stack label { display: grid; gap: 4px; font-size: 0.85rem; color: var(--text-secondary); }
  input[type=text], input[type=password] {
    font: inherit; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--baseline);
    background: var(--page); color: var(--text-primary);
  }
  pre { background: var(--raised); padding: 10px 12px; border-radius: 8px; overflow-x: auto; }
  dialog { border: 1px solid var(--border); border-radius: 12px; background: var(--surface);
           color: var(--text-primary); max-width: 460px; }
  dialog::backdrop { background: rgba(0,0,0,0.35); }
  dialog .actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px; }
  .chart-wrap { position: relative; height: clamp(240px, 38vh, 360px); }
  .chart-card { margin-bottom: 14px; }
  .chart-card h2 { font-size: 0.98rem; font-weight: 650; margin: 0 0 2px; }
  .empty { position: absolute; inset: 0; display: none; align-items: center;
           justify-content: center; color: var(--muted); text-align: center; padding: 0 20px; }
  .center { max-width: 420px; margin: 48px auto; padding: 0 16px; }
"""


def _top(login: str | None, sign_in: bool, csrf: str | None) -> str:
    if sign_in and login:
        right = (f'<span>signed in as <b>{escape(login)}</b></span>'
                 f'<form method="post" action="/signout">'
                 f'<input type="hidden" name="csrf_token" value="{escape(csrf or "")}">'
                 f'<button type="submit">Sign out</button></form>')
    elif sign_in:
        right = ""
    else:
        right = ('<span>Sign-in off</span>'
                 '<a href="/admin-account">Create Admin Account</a>')
    return (f'<header class="top"><span class="brand">OumiGo console</span>'
            f'<span class="muted">{escape(__version__)}</span>'
            f'<div class="right">{right}</div></header>')


def page(title: str, body: str, *, active: str | None = None, login: str | None = None,
         sign_in: bool = False, csrf: str | None = None, head: str = "",
         script: str = "") -> str:
    """A full page: the top bar, the topics (when `active` is set) and `body`."""
    if active is None:
        content = f'<div class="center">{body}</div>'
    else:
        links = "".join(
            f'<a href="{href}"{" class=active" if key == active else ""}>{escape(label)}</a>'
            for key, href, label in TOPICS)
        content = f'<div class="layout"><nav class="side">{links}</nav><main>{body}</main></div>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="csrf-token" content="{escape(csrf or '')}">
<title>{escape(title)} · OumiGo console</title>
{head}
<style>{_CSS}</style>
</head>
<body>
{_top(login, sign_in, csrf)}
{content}
{f"<script>{script}</script>" if script else ""}
</body>
</html>"""


def _errors(problems: list[str]) -> str:
    if not problems:
        return ""
    if len(problems) == 1:
        return f'<div class="error">{escape(problems[0])}</div>'
    items = "".join(f"<li>{escape(p)}</li>" for p in problems)
    return f'<div class="error">Please fix:<ul>{items}</ul></div>'


def signin_body(csrf: str, login: str = "", problems: list[str] | None = None) -> str:
    return f"""<h1>Sign in</h1>
<p class="sub">OumiGo console</p>
{_errors(problems or [])}
<form class="stack card" method="post" action="/signin">
  <input type="hidden" name="csrf_token" value="{escape(csrf)}">
  <label>Login name<input type="text" name="login" value="{escape(login)}"
         autocomplete="username" autofocus required></label>
  <label>Password<input type="password" name="password"
         autocomplete="current-password" required></label>
  <div><button class="primary" type="submit">Sign in</button></div>
</form>"""


def admin_form_body(csrf: str, env_file: str, login: str = "",
                    problems: list[str] | None = None) -> str:
    return f"""<h1>Create Admin Account</h1>
<p class="sub">Sign-in is off. This creates the console's admin account and writes it to
<span class="mono">{escape(env_file)}</span>. Sign-in turns on at once.</p>
{_errors(problems or [])}
<form class="stack card" method="post" action="/admin-account">
  <input type="hidden" name="csrf_token" value="{escape(csrf)}">
  <label>Login name<input type="text" name="login" value="{escape(login)}"
         autocomplete="username" required></label>
  <label>Password<input type="password" name="password" autocomplete="new-password"
         required></label>
  <label>Password again<input type="password" name="password2" autocomplete="new-password"
         required></label>
  <p class="muted" style="margin:0">Login name: 3 to 40 lowercase letters, digits, '.', '_'
  or '-'. Password: at least 10 printable ASCII characters (no spaces), with a lowercase
  letter, an uppercase letter, a digit and a symbol.</p>
  <div><button class="primary" type="submit">Create account</button></div>
</form>"""


def admin_manual_body(reason: str, env_file: str | None) -> str:
    target = escape(env_file) if env_file else "the manager's .env file"
    return f"""<h1>Create Admin Account</h1>
<div class="notice">{escape(reason)}</div>
<div class="card">
<p style="margin-top:0">Create the account on the manager host:</p>
<ol>
  <li>Make the password hash: <pre>oumigo manager admin-password</pre></li>
  <li>Add the login and the printed hash to <span class="mono">{target}</span>
      (no comments on the same lines):
      <pre>MG_ADMIN_USER=&lt;login&gt;
MG_ADMIN_PASSWORD_HASH=scrypt:…</pre></li>
  <li>Restart the manager, e.g. <pre>sudo systemctl restart oumigo-manager</pre></li>
</ol>
</div>"""


def workers_body(control_note: str | None) -> str:
    note = f'<div class="notice">{escape(control_note)}</div>' if control_note else ""
    return f"""<h1>Workers</h1>
<p class="sub">Registered workers, as <span class="mono">workers</span> in the manager console
shows them. <span class="status" id="status">loading…</span></p>
{note}
<div class="card scroll"><table>
<thead><tr><th>WORKER</th><th>HASH ID</th><th>IP</th><th>PORT</th><th>GPU#0</th>
<th>BACKEND</th><th>TRANSPORT</th><th>STATE</th><th>MODEL_NAME</th><th>TASK</th>
<th>START (UTC)</th><th>LAST SEEN</th><th></th></tr></thead>
<tbody id="rows"><tr><td colspan="13" class="muted">loading…</td></tr></tbody>
</table></div>
<dialog id="confirm">
  <p id="confirm-text" style="margin-top:0"></p>
  <p class="muted" style="margin:0">It stops taking new requests, finishes the ones in
  progress, then shuts down. A container platform may restart it.</p>
  <div class="actions">
    <button type="button" id="confirm-cancel">Cancel</button>
    <button type="button" class="primary" id="confirm-ok">Close worker</button>
  </div>
</dialog>"""


WORKERS_JS = """
(function () {
  "use strict";
  var csrf = document.querySelector('meta[name="csrf-token"]').content;
  var dialog = document.getElementById("confirm"), pending = null;

  function cell(text, cls) {
    var td = document.createElement("td");
    td.textContent = text == null ? "-" : text;
    if (cls) td.className = cls;
    return td;
  }
  function render(data) {
    var body = document.getElementById("rows");
    body.textContent = "";
    if (!data.workers.length) {
      var tr = document.createElement("tr");
      var td = cell("No workers registered yet.", "muted");
      td.colSpan = 13; tr.appendChild(td); body.appendChild(tr);
    }
    data.workers.forEach(function (w) {
      var tr = document.createElement("tr");
      [[w.name], [w.node_id, "mono"], [w.address], [w.port], [w.gpu], [w.backend],
       [w.transport], [w.state], [w.model], [w.task], [w.start], [w.last_seen]
      ].forEach(function (c) { tr.appendChild(cell(c[0], c[1])); });
      var td = document.createElement("td");
      if (w.closing) {
        td.appendChild(document.createTextNode("Closing…"));
        td.className = "muted";
      } else {
        var b = document.createElement("button");
        b.textContent = "Close"; b.className = "danger";
        b.disabled = !w.closable;
        if (!data.control) b.title = data.control_note || "";
        b.addEventListener("click", function () { ask(w); });
        td.appendChild(b);
      }
      tr.appendChild(td);
      body.appendChild(tr);
    });
    document.getElementById("status").textContent = "updated " + data.generated_at + " UTC";
  }
  function poll() {
    fetch("/api/workers", { cache: "no-store" })
      .then(function (r) {
        if (r.status === 401) { location.href = "/signin"; throw new Error("signed out"); }
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then(render)
      .catch(function () {
        document.getElementById("status").textContent = "reconnecting… (last update failed)";
      });
  }
  function ask(w) {
    pending = w;
    document.getElementById("confirm-text").textContent =
      "Close " + w.name + " (" + w.node_id + ")?";
    dialog.showModal();
  }
  document.getElementById("confirm-cancel").addEventListener("click", function () {
    pending = null; dialog.close();
  });
  document.getElementById("confirm-ok").addEventListener("click", function () {
    var w = pending; pending = null; dialog.close();
    if (!w) return;
    fetch("/api/workers/" + encodeURIComponent(w.node_id) + "/close", {
      method: "POST", headers: { "X-CSRF-Token": csrf }
    }).then(function (r) {
      if (r.status === 401) { location.href = "/signin"; return; }
      return r.json().then(function (d) {
        if (!r.ok) alert(d.detail || ("Close failed (" + r.status + ")"));
        poll();
      });
    });
  });
  poll();
  setInterval(poll, 5000);
})();
"""


# Chart views: one card per range. (key, title, note, refresh seconds)
GPU_CHARTS = (
    ("60m", "Past 60 Minutes", "5 s grid · 6-sample moving average", 5),
    ("48h", "Past 48 Hours", "5-minute grid · mean of each 5 minutes", 60),
    ("10d", "Past 10 Days", "30-minute grid · mean of each 30 minutes", 60),
    ("45d", "Past 45 Days", "2-hour grid · mean of each 2 hours", 60),
)
TOKEN_RATE_CHARTS = (
    ("60m", "Past 60 Minutes", "5 s grid · MT / min · 6-sample moving average", 5),
    ("48h", "Past 48 Hours", "5-minute grid · MT / min", 60),
    ("10d", "Past 10 Days", "30-minute grid · MT / min", 60),
    ("45d", "Past 45 Days", "2-hour grid · MT / min", 60),
)
TOKEN_CUMULATIVE_CHARTS = (
    ("60m", "Past 60 Minutes", "5 s grid · MT, running total from the left edge", 5),
    ("48h", "Past 48 Hours", "5-minute grid · MT, running total from the left edge", 60),
    ("10d", "Past 10 Days", "30-minute grid · MT, running total from the left edge", 60),
    ("45d", "Past 45 Days", "2-hour grid · MT, running total from the left edge", 60),
)
GPU_INTRO = ("Average <span class=\"mono\">gpu:#N_util_pct</span> per worker (the mean "
             "across its GPUs).")
TOKEN_INTRO = ("Tokens per worker from vLLM's counters: <b>generated</b> (solid) and "
               "<b>input</b> (dashed), plus the total of all workers. MT = million tokens.")


def chart_body(title: str, intro: str, charts) -> str:
    """A view of stacked charts, one card per range; `intro` is trusted HTML."""
    cards = "".join(f"""
<section class="card chart-card">
  <h2>{escape(t)}</h2>
  <p class="sub">{escape(note)} · times in UTC ·
     <span class="status" id="status-{key}">loading…</span></p>
  <div class="chart-wrap">
    <canvas id="chart-{key}"></canvas>
    <div class="empty" id="empty-{key}">Nothing recorded for this period yet.</div>
  </div>
</section>""" for key, t, note, _ in charts)
    return f"""<h1>{escape(title)}</h1>
<p class="sub">{intro} The newest point of each chart covers the period so far.</p>
{cards}"""


def chart_script(endpoint: str, charts, unit: str, percent: bool) -> str:
    """The view's script: poll `endpoint` per range and draw it."""
    config = {"endpoint": endpoint, "unit": unit, "percent": percent,
              "charts": [[key, every] for key, _, _, every in charts]}
    return CHART_JS.replace("__CONFIG__", json.dumps(config))


CHART_HEAD = f'<script src="{CHARTJS_SRC}"></script>'

CHART_JS = """
(function () {
  "use strict";
  var CONFIG = __CONFIG__;
  // Categorical palette in fixed slot order; more than 8 workers cycle.
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
  function fmt(v) {
    if (v == null) return "—";
    if (CONFIG.percent) return v + "%";
    return (v === 0 ? "0" : String(Number(v.toPrecision(3)))) + " " + CONFIG.unit;
  }
  function buildOptions() {
    var muted = ink("--muted"), grid = ink("--grid"), text = ink("--text-secondary");
    var y = { beginAtZero: true, border: { display: false }, grid: { color: grid },
              ticks: { color: muted, callback: function (v) {
                return CONFIG.percent ? v + "%" : String(Number(v.toPrecision(3)));
              } } };
    if (CONFIG.percent) { y.min = 0; y.max = 100; }
    else { y.title = { display: true, text: CONFIG.unit, color: muted }; }
    return {
      responsive: true, maintainAspectRatio: false, animation: false, spanGaps: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: y,
        x: { border: { color: ink("--baseline") },
             ticks: { color: muted, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
             grid: { display: false } }
      },
      plugins: {
        legend: { labels: { color: text, boxWidth: 12, boxHeight: 2 } },
        tooltip: { callbacks: { label: function (c) {
          return c.dataset.label + ": " + fmt(c.parsed.y);
        } } }
      }
    };
  }
  function toDatasets(series, sparse) {
    var pal = PALETTE[isDark() ? "dark" : "light"], workers = {}, next = 0;
    return series.map(function (s, i) {
      var color, width = 2;
      if (s.total) { color = ink("--text-primary"); width = 3; }
      else if (s.worker) {
        if (!(s.worker in workers)) workers[s.worker] = next++;
        color = pal[workers[s.worker] % pal.length];
      } else { color = pal[i % pal.length]; }
      return { label: s.label, data: s.data, borderColor: color, backgroundColor: color,
               borderWidth: width, borderDash: s.kind === "input" ? [6, 4] : [],
               pointRadius: sparse ? 1.5 : 0, pointHoverRadius: 4, tension: 0.25,
               spanGaps: false };
    });
  }
  function track(spec) {
    var key = spec[0], every = spec[1] * 1000, chart = null;
    function render(p) {
      var present = p.series.some(function (s) {
        return s.data.some(function (v) { return v != null; });
      });
      document.getElementById("empty-" + key).style.display = present ? "none" : "flex";
      var datasets = toDatasets(p.series, key !== "60m");
      if (!chart) {
        chart = new Chart(document.getElementById("chart-" + key).getContext("2d"), {
          type: "line", data: { labels: p.labels, datasets: datasets },
          options: buildOptions()
        });
      } else {
        chart.data.labels = p.labels;
        chart.data.datasets = datasets;
        chart.options = buildOptions();
        chart.update();
      }
      var workers = {};
      p.series.forEach(function (s) { if (!s.total) workers[s.node_id || s.label] = 1; });
      var n = Object.keys(workers).length;
      document.getElementById("status-" + key).textContent =
        "updated " + p.generated_at + " · " + n + " worker" + (n === 1 ? "" : "s");
    }
    function poll() {
      var sep = CONFIG.endpoint.indexOf("?") < 0 ? "?" : "&";
      fetch(CONFIG.endpoint + sep + "range=" + key, { cache: "no-store" })
        .then(function (r) {
          if (r.status === 401) { location.href = "/signin"; throw new Error("signed out"); }
          if (!r.ok) throw new Error(r.status);
          return r.json();
        })
        .then(render)
        .catch(function () {
          document.getElementById("status-" + key).textContent =
            "reconnecting… (last update failed)";
        });
    }
    poll();
    setInterval(poll, every);
    return poll;
  }
  var polls = CONFIG.charts.map(track);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      polls.forEach(function (p) { p(); });
    });
  }
})();
"""
