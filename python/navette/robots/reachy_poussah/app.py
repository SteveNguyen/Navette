"""The robot loop as a Reachy Mini app: the SDK's ReachyMiniApp serves the status page (navette.runtime.page
routes on its settings_app, at custom_app_url) and calls run() with its ReachyMini.

NOT INSTALLED by this package. To install it, a pyproject needs the entry point
`[project.entry-points."reachy_mini_apps"] navette = "navette.robots.reachy_poussah.app:NavetteApp"`, the
robot config with its model and calibration files on the robot, and bleak importable.

As an app it is NOT a dry run: START drives the base. The config is NAVETTE_ROBOT (default
robots/reachy_poussah.toml in the working directory), the run folders go under NAVETTE_OUT.
Robot only (imports reachy_mini at the top); run by hand with `python -m navette.robots.reachy_poussah.app`.
"""
import os
import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from navette.runtime import cli


class NavetteApp(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8090"  # the config's page.port
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        if self.settings_app is None:
            raise RuntimeError("settings_app is not initialized")
        args = cli.build_parser().parse_args(["--robot", os.environ.get("NAVETTE_ROBOT", "robots/reachy_poussah.toml"),
                                              "--out", os.environ.get("NAVETTE_OUT", "/home/pollen/navette_runs")])
        cli.run_robot(args, stop_event, self.settings_app, log=self.logger.info, mini=reachy_mini)


if __name__ == "__main__":
    app = NavetteApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
