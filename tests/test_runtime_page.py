"""Status page endpoints (navette.runtime.page); skipped where fastapi / httpx are not installed."""
import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from navette.runtime import controller as lr  # noqa: E402
from navette.runtime import page  # noqa: E402
from navette.runtime.loop import Joystick, NavOut, Status  # noqa: E402
from navette.testing.fakes import FakeBase, FakeWorld, ManualClock  # noqa: E402


def client(base=None, **kw):
    ctrl = lr.Controller(base, 0.15, 0.5, dry_run=True, log=lambda m: None, **kw)
    status = Status()
    app = page.new_app()
    page.add_routes(app, ctrl, status)
    return TestClient(app), ctrl, status


def test_page_and_status():
    c, ctrl, status = client()
    r = c.get("/")
    assert "START" in r.text and r.headers["cache-control"] == "no-store"  # never an earlier run's page
    s = c.get("/api/status").json()
    assert s["state"] == "IDLE" and s["v_max"] == 0.15 and s["dry_run"] and s["base"] is None
    assert c.get("/api/preview.jpg").status_code == 503
    status.preview = b"\xff\xd8jpeg"
    r = c.get("/api/preview.jpg")
    assert r.status_code == 200 and r.content == b"\xff\xd8jpeg" and r.headers["content-type"] == "image/jpeg"


def test_poll_period_follows_the_page_timeout():
    """The page's poll is the heartbeat of the page dead-man: with a fixed 0.5 s, a page.timeout_s under it
    halted every run."""
    assert page.poll_period(2.0) == 0.5 and page.poll_period(0.0) == 0.5 and page.poll_period(30.0) == 0.5
    assert page.poll_period(0.4) == pytest.approx(0.1)
    html = page.page_html(2.0, 2.0)  # the reference robot's config: every 0.5 s, a preview every 4th poll
    assert "setInterval(poll, 500)" in html and "n++ % 4 === 0" in html and "__" not in html
    html = page.page_html(0.4, 2.0)
    assert "setInterval(poll, 100)" in html and "n++ % 20 === 0" in html
    ctrl = lr.Controller(None, 0.15, 0.5, log=lambda m: None)
    app = page.new_app()
    page.add_routes(app, ctrl, Status(), timeout_s=0.8, preview_s=1.0)
    assert "setInterval(poll, 200)" in TestClient(app).get("/").text


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


def teleop_client(**kw):
    ctrl = lr.Controller(None, 0.15, 0.5, dry_run=True, log=lambda m: None, **kw)
    status, joystick = Status("teleop"), Joystick()
    app = page.new_app()
    page.add_routes(app, ctrl, status, joystick=joystick)
    return TestClient(app), ctrl, status, joystick


def test_teleop_page_has_the_joystick_only_in_a_teleop_run():
    c, *_ = client()
    assert "const TELEOP = false" in c.get("/").text
    c, *_ = teleop_client()
    html = c.get("/").text
    assert "const TELEOP = true" in html and 'id="pad"' in html and "__" not in html
    assert "setInterval(send, 100)" in html  # ~10 Hz while the thumb is down
    assert "user-scalable=no" in html and "touch-action: none" in html  # a phone: no zoom, no scroll on the pad


def test_teleop_endpoint_sets_the_joystick_and_shows_the_allowed_speed():
    c, ctrl, status, joystick = teleop_client()
    assert joystick.get() is None
    assert c.post("/api/teleop", json={"v": 0.12, "w": -0.3}).json()["ok"]
    v, w, t = joystick.get()
    assert (v, w) == (0.12, -0.3) and t == pytest.approx(time.monotonic(), abs=1.0)
    assert c.post("/api/teleop", json={"v": 0, "w": 0}).json()["ok"] and joystick.get()[:2] == (0.0, 0.0)
    # the status shows the behaviour, what was asked and what the collision monitor allowed
    status.v_req, status.w_req, status.v_ok, status.w_ok = 0.15, 0.0, 0.0, 0.0
    s = c.get("/api/status").json()
    assert s["behaviour"] == "teleop" and (s["v_req"], s["v_ok"]) == (0.15, 0.0)


@pytest.mark.parametrize("body", [{"v": "0.1", "w": 0}, {"v": 0.1}, {"v": float("nan"), "w": 0}, {"v": True, "w": 0},
                                  [0.1, 0.0], "x"])
def test_teleop_endpoint_refuses_bad_values(body):
    c, _, _, joystick = teleop_client()
    # json.dumps writes NaN as is (not JSON, but Python's parser takes it): refused as not finite
    r = c.post("/api/teleop", content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and joystick.get() is None


def test_teleop_endpoint_needs_json_and_a_teleop_run():
    c, _, _, joystick = teleop_client()
    r = c.post("/api/teleop", content='{"v": 0.1, "w": 0}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415 and joystick.get() is None
    c, *_ = client()
    assert c.post("/api/teleop", json={"v": 0.1, "w": 0.0}).status_code == 409


def test_teleop_post_is_a_page_heartbeat():
    clock = ManualClock()
    c, ctrl, _, _ = teleop_client(clock=clock, page_timeout_s=2.0)
    assert c.post("/api/start", json={}).json()["ok"]
    for k in range(60):  # 3 s, no status poll, the joystick posting every 0.5 s
        clock.t += 0.05
        ctrl.set_nav(NavOut(0.1, 0.0, 0.1, 0.0, "TELEOP", False), clock.t, ctrl.run)
        ctrl.tick()
        if k % 10 == 0:
            c.post("/api/teleop", json={"v": 0.1, "w": 0.0})
    assert ctrl.state == "DRIVING"


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
