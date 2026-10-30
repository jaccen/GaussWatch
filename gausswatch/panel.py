"""GaussWatch zero-dependency web panel.

Standard library only (http.server + json). One process, one page, no npm:

    from gausswatch.collector import TrainingMonitor
    from gausswatch.panel import PanelServer, serve

    m = TrainingMonitor()
    # ... feed training events ...

    serve(m, port=8420)            # blocks until Ctrl+C
    # or, embedded in an existing script / notebook:
    srv = PanelServer(m)           # port=0 picks a free port
    srv.start(block=False)
    print(f"http://127.0.0.1:{srv.port}")

Endpoints:
    GET /           single-page dashboard (polls /api/state every 2 s)
    GET /api/state  {"stats": {...}, "alerts": [{rule, severity, message,
                     source, first_seen}, ...]}
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collector import TrainingMonitor

DEFAULT_PORT = 8420

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GaussWatch</title>
<style>
  :root {
    --bg: #0A1628;         /* panel night blue */
    --panel: #101E33;
    --line: #1C2F4A;
    --text: #D7E3F4;
    --dim:  #7B8CA6;
    --acc:  #00D4AA;       /* teal accent */
    --crit: #FF5470;
    --warn: #FFB454;
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--text);
         font: 14px/1.5 "Segoe UI", system-ui, sans-serif; padding: 24px; }
  header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 18px; }
  h1 { font-size: 20px; letter-spacing: .04em; color: var(--acc); }
  .meta { color: var(--dim); }
  .badge { display: inline-block; padding: 1px 9px; border-radius: 10px;
           font-size: 12px; margin-left: 8px; }
  .b-crit { background: rgba(255,84,112,.15); color: var(--crit); }
  .b-warn { background: rgba(255,180,84,.15); color: var(--warn); }
  table { width: 100%; border-collapse: collapse; background: var(--panel);
          border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 600; font-size: 12px; text-transform: uppercase;
       letter-spacing: .06em; }
  tr:last-child td { border-bottom: none; }
  td.rule { color: var(--acc); font-family: Consolas, monospace; white-space: nowrap; }
  td.time { color: var(--dim); white-space: nowrap; }
  td.src  { color: var(--dim); font-size: 12px; }
  .sev-critical { color: var(--crit); font-weight: 600; }
  .sev-warning  { color: var(--warn); font-weight: 600; }
  .empty { color: var(--dim); text-align: center; padding: 32px 0; }
  .foot  { color: var(--dim); font-size: 12px; margin-top: 14px; }
</style>
</head>
<body>
<header>
  <h1>GaussWatch</h1>
  <span class="meta">step <span id="step">-</span></span>
  <span class="meta">
    <span class="badge b-crit" id="n-crit">critical 0</span>
    <span class="badge b-warn" id="n-warn">warning 0</span>
  </span>
</header>
<table>
  <thead><tr><th>rule</th><th>severity</th><th>message</th><th>source</th><th>first seen</th></tr></thead>
  <tbody id="rows"><tr><td colspan="5" class="empty">loading&hellip;</td></tr></tbody>
</table>
<p class="foot">polling /api/state every 2 s &middot; no data leaves this machine</p>
<script>
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function tick() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    document.getElementById('step').textContent = d.stats.step;
    document.getElementById('n-crit').textContent = 'critical ' + d.stats.n_critical;
    document.getElementById('n-warn').textContent = 'warning ' + d.stats.n_warning;
    const rows = document.getElementById('rows');
    if (!d.alerts.length) {
      rows.innerHTML = '<tr><td colspan="5" class="empty">no alerts yet — training looks clean</td></tr>';
      return;
    }
    rows.innerHTML = d.alerts.map(a =>
      '<tr><td class="rule">' + esc(a.rule) + '</td>' +
      '<td class="sev-' + a.severity + '">' + esc(a.severity) + '</td>' +
      '<td>' + esc(a.message) + '</td>' +
      '<td class="src">' + esc(a.source) + '</td>' +
      '<td class="time">' + new Date(a.first_seen * 1000).toLocaleTimeString() + '</td></tr>'
    ).join('');
  } catch (e) { /* transient: keep polling */ }
}
setInterval(tick, 2000);
tick();
</script>
</body>
</html>"""


class PanelServer:
    """Serve the GaussWatch dashboard for one TrainingMonitor. Thread-safe."""

    def __init__(self, monitor: TrainingMonitor, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT):
        mon = monitor  # captured by the handler closure below

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/api/state"):
                    body = json.dumps({
                        "stats": mon.stats(),
                        "alerts": mon.alert_records(),
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = _PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *args):  # silence default stderr access log
                pass

        self._http = ThreadingHTTPServer((host, port), Handler)

    @property
    def port(self) -> int:
        return self._http.server_address[1]

    def start(self, block: bool = False):
        """block=True serves on the calling thread (Ctrl+C stops it);
        block=False runs a daemon thread and returns immediately."""
        if block:
            try:
                self._http.serve_forever()
            finally:
                self.shutdown()
        else:
            threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def shutdown(self):
        self._http.shutdown()
        self._http.server_close()


def serve(monitor: TrainingMonitor, host: str = "127.0.0.1",
          port: int = DEFAULT_PORT):
    """Convenience wrapper: build, print the URL, serve until Ctrl+C."""
    srv = PanelServer(monitor, host=host, port=port)
    print(f"GaussWatch panel: http://{host}:{srv.port}  (Ctrl+C to stop)")
    srv.start(block=True)
