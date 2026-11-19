"""Status page of the robot loop (navette.runtime.loop): START, STOP, a status line and a camera + map
preview; in a teleop run, a touch joystick. Served by the robot over WiFi, by the CLI (uvicorn on page.port) or
by the robot's own app server.

  GET  /              the page (polls /api/status every poll_period(page.timeout_s), reloads
                      /api/preview.jpg every page.preview_s)
  GET  /api/status    controller (state, last watchdog / stop reason, v / w sent, speed cap, dry run, base
                      state) + loop (Hz, IMU source, behaviour and its state, camera height / pitch, the last
                      step's request v_req / w_req and what the collision monitor allowed of it, v_ok / w_ok).
                      Every call is a heartbeat of the controller's page dead-man.
  POST /api/start     START: {"ok", "status"}   all three need Content-Type: application/json:
  POST /api/stop      STOP:  {"ok", "status"}   415 otherwise
  POST /api/teleop    {"v": m/s, "w": rad/s}: the operator's request (teleop runs only: 409 otherwise; 400 if v
                      or w is not a finite number). Also a heartbeat. The navigator holds it for teleop.timeout_s
                      (operator dead-man) and sends it through the collision monitor like any request.
The page has no authentication. Requiring JSON means another web page open on a phone on the same WiFi cannot
send them with a plain form or a "simple" fetch: a cross-origin JSON POST needs a CORS preflight, which this
server does not answer (no CORS middleware: keep it that way). Our own page is same-origin, so its buttons work.
  GET  /api/preview.jpg  latest camera | map view, 503 before the first one

Joystick (teleop runs): one thumb, up = forward, sideways = turn (right = clockwise), scaled to the
controller's caps (v_max, w_max of /api/status); sent every JOY_MS while the thumb is down, (0, 0) once on
release. Shown: what the joystick sends and what the robot is allowed (v_ok, w_ok), "HELD" when the collision
monitor holds a forward request.

fastapi is imported here only (robot side; it is in the dev dependency group for the tests).
"""
import math

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Navette</title>
<style>
 html, body { touch-action: manipulation; }  /* no zoom on a double tap */
 body { font-family: sans-serif; margin: 12px; background: #111; color: #eee; }
 button { font-size: 1.6em; width: 48%; padding: 18px 0; border: 0; border-radius: 8px; color: #fff; }
 #start { background: #2a7a2a; } #stop { background: #b02020; }
 #status { font-family: monospace; white-space: pre-wrap; margin: 10px 0; font-size: 0.95em; }
 img { width: 100%; }
 #teleop { display: none; text-align: center; }
 #pad { position: relative; width: min(80vw, 320px); height: min(80vw, 320px); margin: 10px auto;
        border-radius: 50%; background: #2a2a2a; border: 2px solid #555; touch-action: none;
        user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; }
 #knob { position: absolute; left: 35%; top: 35%; width: 30%; height: 30%; border-radius: 50%;
         background: #3a6ab0; pointer-events: none; }
 #speeds { font-family: monospace; font-size: 1.1em; white-space: pre; }
 .held { color: #ff6060; font-weight: bold; }
</style></head><body>
<div><button id="start" onclick="post('start')">START</button>
<button id="stop" onclick="post('stop')">STOP</button></div>
<div id="teleop"><div id="pad"><div id="knob"></div></div><div id="speeds">...</div></div>
<div id="status">...</div>
<img id="preview" alt="no preview yet">
<script>
const TELEOP = __TELEOP__;
async function post(what, body) {
  const r = await fetch('/api/' + what, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify(body || {})});
  const j = await r.json();
  if (what !== 'teleop') document.getElementById('status').textContent = j.status;
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
// joystick: (jx, jy) in the unit disk, scaled to the controller's caps from the last status
let jx = 0, jy = 0, vMax = 0.15, wMax = 0.5, active = null, timer = null, busy = false, last = null;
const pad = document.getElementById('pad'), knob = document.getElementById('knob');
function request() {
  const dead = (a) => Math.abs(a) < 0.1 ? 0 : a;  // a thumb is never exactly centred
  return {v: dead(jy) * vMax, w: -dead(jx) * wMax};
}
function show() {
  const r = pad.getBoundingClientRect().width / 2, q = request();
  knob.style.transform = `translate(${jx * r}px, ${-jy * r}px)`;
  const held = last && last.v_req > 0.01 && last.v_ok < 0.5 * last.v_req;
  document.getElementById('speeds').innerHTML =
    `joystick v ${q.v.toFixed(2)} w ${q.w.toFixed(2)}\\n` +
    (last ? `allowed  v ${last.v_ok.toFixed(2)} w ${last.w_ok.toFixed(2)}` : 'allowed  ...') +
    (held ? '  <span class="held">HELD</span>' : '');
}
async function send(force) {
  if (busy && !force) return;  // the previous one still on its way: skip, do not queue
  busy = true;
  try { await post('teleop', request()); } catch (e) {} finally { busy = false; }
}
function move(e) {
  const r = pad.getBoundingClientRect(), h = r.width / 2;
  let x = (e.clientX - r.left - h) / h, y = -(e.clientY - r.top - h) / h;
  const n = Math.hypot(x, y);
  if (n > 1) { x /= n; y /= n; }
  jx = x; jy = y; show();
}
function release(e) {
  if (active === null || (e && e.pointerId !== active)) return;
  active = null; jx = 0; jy = 0;
  clearInterval(timer); timer = null;
  show(); send(true);
}
if (TELEOP) {
  document.getElementById('teleop').style.display = 'block';
  pad.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    active = e.pointerId; pad.setPointerCapture(e.pointerId); move(e); send(true);
    if (timer === null) timer = setInterval(send, __JOY_MS__);
  });
  pad.addEventListener('pointermove', (e) => { if (e.pointerId === active) move(e); });
  pad.addEventListener('pointerup', release);
  pad.addEventListener('pointercancel', release);
  document.addEventListener('visibilitychange', () => { if (document.hidden) release(); });
  show();
}
// the preview only every few polls (page.preview_s): on a robot whose WiFi and Bluetooth share one radio, a
// JPEG every 0.5 s coincided with a 0.9 s stall of the base's telemetry and the page losing its connection
let n = 0;
async function poll() {
  try {
    const s = await (await fetch('/api/status')).json();
    document.getElementById('status').textContent = fmt(s);
    vMax = s.v_max; wMax = s.w_max; last = s;
    if (TELEOP) show();
  } catch (e) { document.getElementById('status').textContent = 'no connection'; }
  if (n++ % __PREVIEW_EVERY__ === 0) document.getElementById('preview').src = '/api/preview.jpg?' + Date.now();
}
setInterval(poll, __POLL_MS__); poll();
</script></body></html>
"""


POLL_S = 0.5  # the page's poll period, unless the page dead-man needs it shorter
JOY_MS = 100  # joystick send period while the thumb is down (~10 Hz; the operator's dead-man is teleop.timeout_s)


def poll_period(timeout_s):
    """Poll period (s) of the page for a page dead-man of timeout_s (the config's page.timeout_s; 0 = off):
    POLL_S, or a quarter of the timeout if that is shorter, so that the dead-man sees at least three polls
    in its time whatever the config says (a timeout under the fixed 0.5 s halted every run)."""
    return min(POLL_S, timeout_s / 4) if timeout_s > 0 else POLL_S


def page_html(timeout_s=2.0, preview_s=2.0, teleop=False):
    """The page for a config's page.timeout_s and page.preview_s; teleop: with the joystick."""
    poll_s = poll_period(timeout_s)
    return PAGE.replace("__POLL_MS__", str(round(poll_s * 1000))).replace(
        "__PREVIEW_EVERY__", str(max(1, round(preview_s / poll_s)))).replace(
        "__TELEOP__", "true" if teleop else "false").replace("__JOY_MS__", str(JOY_MS))


def add_routes(app, ctrl, status, timeout_s=2.0, preview_s=2.0, joystick=None):
    """Page routes on a FastAPI app; ctrl: runtime.controller.Controller, status: runtime.loop.Status;
    timeout_s, preview_s: the config's page.timeout_s and page.preview_s (the page's poll and preview
    periods follow them); joystick: the teleop run's runtime.loop.Joystick, None in other runs (no joystick
    on the page, /api/teleop refused)."""
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
        # no-store: the page differs between runs (joystick or not, poll period); a phone kept showing an earlier
        # run's page without the joystick until reloaded
        return HTMLResponse(page_html(timeout_s, preview_s, joystick is not None),
                            headers={"Cache-Control": "no-store"})

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

    @app.post("/api/teleop")
    async def teleop(request: Request):
        if (bad := not_json(request)) is not None:
            return bad
        if joystick is None:
            return JSONResponse({"ok": False, "status": "not a teleop run"}, status_code=409)
        try:
            body = await request.json()
            v, w = body["v"], body["w"]
        except (ValueError, TypeError, KeyError):
            return JSONResponse({"ok": False, "status": 'expected {"v": number, "w": number}'}, status_code=400)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (v, w)):
            return JSONResponse({"ok": False, "status": "v and w must be finite numbers"}, status_code=400)
        ctrl.page_alive()  # the joystick page is alive
        joystick.set(v, w)
        return {"ok": True, "status": "ok"}

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
