"""Status page endpoints (navette.runtime.page); skipped where fastapi / httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from navette.runtime import controller as lr  # noqa: E402
from navette.runtime import page  # noqa: E402
from navette.runtime.loop import NavOut, Status  # noqa: E402
from navette.testing.fakes import FakeBase, FakeWorld, ManualClock  # noqa: E402


def client(base=None, **kw):
    ctrl = lr.Controller(base, 0.15, 0.5, dry_run=True, log=lambda m: None, **kw)
    status = Status()
    app = page.new_app()
    page.add_routes(app, ctrl, status)
    return TestClient(app), ctrl, status


def test_page_and_status():
    c, ctrl, status = client()
    assert "START" in c.get("/").text
    s = c.get("/api/status").json()
    assert s["state"] == "IDLE" and s["v_max"] == 0.15 and s["dry_run"] and s["base"] is None
    assert c.get("/api/preview.jpg").status_code == 503
    status.preview = b"\xff\xd8jpeg"
    r = c.get("/api/preview.jpg")
    assert r.status_code == 200 and r.content == b"\xff\xd8jpeg" and r.headers["content-type"] == "image/jpeg"


def test_start_stop():
    c, ctrl, _ = client()
    assert c.post("/api/start", json={}).json() == {"ok": True, "status": "starting"}
    assert c.post("/api/start", json={}).json()["ok"] is False  # not idle
    ctrl.tick()
    ctrl.set_nav(NavOut(0.3, 0.0, 0.3, 0.0, "DRIVE", False), ctrl.clock(), ctrl.run)
    ctrl.tick()
    assert c.get("/api/status").json()["v"] == pytest.approx(0.15)
    assert c.post("/api/stop", json={}).json()["ok"]
    ctrl.tick()
    s = c.get("/api/status").json()
    assert s["state"] == "IDLE" and s["reason"] == "STOP (page)" and s["v"] == 0


@pytest.mark.parametrize("kw", [{}, {"data": {"a": "1"}}, {"content": "{}", "headers": {"Content-Type": "text/plain"}}])
def test_start_stop_need_a_json_request(kw):
    """What a cross-origin page can send without a CORS preflight (no body, a form, text/plain) is refused;
    nothing starts."""
    c, ctrl, _ = client()
    r = c.post("/api/start", **kw)
    assert r.status_code == 415 and ctrl.state == "IDLE"
    assert c.post("/api/stop", **kw).status_code == 415
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    assert "application/json" in c.get("/").text  # our own page sends JSON


def test_status_poll_is_the_page_dead_man_heartbeat_and_shows_the_base():
    clock = ManualClock()
    base = FakeBase(FakeWorld(clock))
    base.low_battery = True
    c, ctrl, _ = client(base, clock=clock, page_timeout_s=2.0)
    assert c.post("/api/start", json={}).json()["ok"]
    for k in range(60):  # 3 s, the page polling every 0.5 s
        clock.t += 0.05
        ctrl.set_nav(NavOut(0.1, 0.0, 0.1, 0.0, "DRIVE", False), clock.t, ctrl.run)
        ctrl.tick()
        if k % 10 == 0:
            s = c.get("/api/status").json()
    assert s["state"] == "DRIVING" and s["base"] == {"state": "DRIVING", "fault": "", "low_battery": True}
    for _ in range(45):  # the page gone for 2.25 s
        clock.t += 0.05
        ctrl.set_nav(NavOut(0.1, 0.0, 0.1, 0.0, "DRIVE", False), clock.t, ctrl.run)
        ctrl.tick()
    s = c.get("/api/status").json()
    assert s["state"] == "IDLE" and "phone page" in s["reason"] and s["base"]["state"] == "STOPPED"
