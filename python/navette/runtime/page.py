"""Status page of the robot loop (navette.runtime.loop): START, STOP, a status line and a camera + map
preview. Served by the robot over WiFi, by the CLI (uvicorn on page.port) or by the robot's own app server.

  GET  /              the page (polls /api/status every poll_period(page.timeout_s), reloads
                      /api/preview.jpg every page.preview_s)
  GET  /api/status    controller (state, last watchdog / stop reason, v / w sent, speed cap, dry run, base
                      state) + loop (Hz, IMU source, behaviour state, camera height / pitch). Every call is a
                      heartbeat of the controller's page dead-man.
  POST /api/start     START: {"ok", "status"}   both need Content-Type: application/json (any JSON body):
  POST /api/stop      STOP:  {"ok", "status"}   415 otherwise
The page has no authentication. Requiring JSON means another web page open on a phone on the same WiFi cannot
send them with a plain form or a "simple" fetch: a cross-origin JSON POST needs a CORS preflight, which this
server does not answer (no CORS middleware: keep it that way). Our own page is same-origin, so its buttons work.
  GET  /api/preview.jpg  latest camera | map view, 503 before the first one

fastapi is imported here only (robot side; the tests skip without it).
"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Navette</title>
<style>
 body { font-family: sans-serif; margin: 12px; background: #111; color: #eee; }
 button { font-size: 1.6em; width: 48%; padding: 18px 0; border: 0; border-radius: 8px; color: #fff; }
 #start { background: #2a7a2a; } #stop { background: #b02020; }
 #status { font-family: monospace; white-space: pre-wrap; margin: 10px 0; font-size: 0.95em; }
 img { width: 100%; }
</style></head><body>
<div><button id="start" onclick="post('start')">START</button>
<button id="stop" onclick="post('stop')">STOP</button></div>
<div id="status">...</div>
<img id="preview" alt="no preview yet">
<script>
async function post(what) {
  const r = await fetch('/api/' + what, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  const j = await r.json();
  document.getElementById('status').textContent = j.status;
}
function fmt(s) {
  const b = s.base;
  const base = b === null ? 'none' : `${b.state}` + (b.fault ? ' ' + b.fault : '') +
    (b.low_battery ? ' LOW BATTERY' : '');
  return `${s.dry_run ? 'DRY RUN  ' : ''}${s.state}  ${s.hz} Hz  ${s.nav_state}\\n` +
    `sent v ${s.v.toFixed(2)} w ${s.w.toFixed(2)}  cap ${s.v_max} m/s\\n` +
    `base ${base}\\nIMU ${s.imu}  h ${s.cam_h}  pitch ${s.pitch}\\n` +
    `${s.reason ? 'last: ' + s.reason : ''} ${s.message}`;
}
// the preview only every few polls (page.preview_s): on a robot whose WiFi and Bluetooth share one radio, a
// JPEG every 0.5 s coincided with a 0.9 s stall of the base's telemetry and the page losing its connection
let n = 0;
async function poll() {
  try {
    const s = await (await fetch('/api/status')).json();
    document.getElementById('status').textContent = fmt(s);
  } catch (e) { document.getElementById('status').textContent = 'no connection'; }
  if (n++ % __PREVIEW_EVERY__ === 0) document.getElementById('preview').src = '/api/preview.jpg?' + Date.now();
}
setInterval(poll, __POLL_MS__); poll();
</script></body></html>
"""


POLL_S = 0.5  # the page's poll period, unless the page dead-man needs it shorter


def poll_period(timeout_s):
    """Poll period (s) of the page for a page dead-man of timeout_s (the config's page.timeout_s; 0 = off):
    POLL_S, or a quarter of the timeout if that is shorter, so that the dead-man sees at least three polls
    in its time whatever the config says (a timeout under the fixed 0.5 s halted every run)."""
    return min(POLL_S, timeout_s / 4) if timeout_s > 0 else POLL_S


def page_html(timeout_s=2.0, preview_s=2.0):
    """The page for a config's page.timeout_s and page.preview_s."""
    poll_s = poll_period(timeout_s)
    return PAGE.replace("__POLL_MS__", str(round(poll_s * 1000))).replace(
        "__PREVIEW_EVERY__", str(max(1, round(preview_s / poll_s))))


def add_routes(app, ctrl, status, timeout_s=2.0, preview_s=2.0):
    """Page routes on a FastAPI app; ctrl: runtime.controller.Controller, status: runtime.loop.Status;
    timeout_s, preview_s: the config's page.timeout_s and page.preview_s (the page's poll and preview
    periods follow them)."""
    from fastapi import Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    def not_json(request):
        """415 unless the request says JSON (see the module docstring: no cross-origin simple requests)."""
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return JSONResponse({"ok": False, "status": "Content-Type must be application/json"}, status_code=415)
        return None

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page_html(timeout_s, preview_s)

    @app.get("/api/status")
    def get_status():
        ctrl.page_alive()  # the page's poll is its heartbeat (Controller page dead-man)
        return {**ctrl.snapshot(), **status.snapshot()}

    @app.post("/api/start")
    def start(request: Request):
        if (bad := not_json(request)) is not None:
            return bad
        ok, msg = ctrl.start()
        return {"ok": ok, "status": msg}

    @app.post("/api/stop")
    def stop(request: Request):
        if (bad := not_json(request)) is not None:
            return bad
        ctrl.stop("STOP (page)")
        return {"ok": True, "status": "stopping"}

    @app.get("/api/preview.jpg")
    def preview():
        jpg = status.preview
        if jpg is None:
            return Response(content="no preview", status_code=503)
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def new_app():
    """An empty FastAPI app (CLI): routes are added by loop.run once the controller exists."""
    from fastapi import FastAPI

    return FastAPI()
