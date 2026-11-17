"""The Cobot Magic twin's Ego Runtime as a process of its own.

Everything the twin knows arrives through its robot port: encoders and gripper states, the
head cameras' RGB-D frames (or labelled detections at phase 0), and the effect of the joint
commands it sends. The belief, the skills and the scenario's scripted mission run here, and the
Ego API serves the belief to the console -- the same shape as the mobile robot's
:class:`~steward.ego.runtime.EgoRuntime`, which is what lets `steward up` put either body behind
a scenario.

A round of the mission ends with the arms parked, and the world decides when the next one
starts: an episode reset (the console's 重置, or loading the scenario again) changes the episode
number of the robot's readings, which the runtime reads as a reboot -- it forgets its belief and
looks again. A reset that
lands in the middle of a round is caught the same way: the round ends however it ends, and the
next one starts at once. The runtime never edits the world; in `steward cobot` the round driver
teleports the tableware home itself, here that is the operator's reset.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from steward.assets.cobot import cobot_magic
from steward.ego.cobot_mission import bowl_onto_plate, carry_to_station, drive_screws, nest_bowls, perceive
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import decode_rgbd
from steward.ego.runtime import RobotLink

MISSIONS: dict[str, Callable] = {
    "bowl-on-plate": bowl_onto_plate,
    "nest-bowls": nest_bowls,
    "carry-to-counter": carry_to_station,
    "drive-screws": drive_screws,
}
# Wide head camera for survey; wrist cameras refine insert/drive aim (see drive_screws).
SURVEY_CAMERA = "wide_camera"
IDLE_LOOK_S = 2.0
RESET_POLL_S = 0.5
TELEMETRY_S = 1.0
RESET_MARGIN_S = 1e-6


class CobotEgoRuntime:
    def __init__(self, link: RobotLink, task: str | None = None, config: MotionConfig | None = None,
                 out_dir: Path | None = None, report: Callable[[dict], None] | None = None,
                 mission_kwargs: dict | None = None):
        if task is not None and task not in MISSIONS:
            raise ValueError(f"unknown mission {task!r}; known: {', '.join(MISSIONS)}")
        link.require_body("cobot-magic")
        self.link = link
        self.task = task
        self.mission_kwargs = dict(mission_kwargs or {})
        self.out_dir = out_dir
        self.report = report or _print_line
        self.estimator = EgoEstimator(robot_model=cobot_magic().to_dict())
        self.skills = CobotSkills(command=link.command, state=self.sensors, config=config or MotionConfig())
        self.rounds = 0
        self.results: list[dict] = []
        self.last_error: str | None = None
        self._stamp = -math.inf
        self._episode: int | None = None
        self._reset = threading.Event()
        self._clock_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._telemetry: threading.Thread | None = None

    # ------------------------------------------------------------ sensing
    def _clock(self, stamp, episode) -> None:
        """Called with every reading's stamp and episode (the caller holds the read lock): a new
        episode, the robot's reboot, changes the episode number or, from a world that numbers
        none, turns the clock back."""
        if episode is not None:
            if self._episode is not None and episode != self._episode:
                self._reset.set()
            self._episode = episode
        if stamp is None:
            return
        if stamp < self._stamp - RESET_MARGIN_S:
            self._reset.set()
        self._stamp = float(stamp)

    def sensors(self) -> dict:
        return self._read(self.link.sensors)

    def proprioception(self) -> dict:
        """The self part of the Ego-State view polls this at frame rate; it draws no perception
        sample, so watching the robot does not change what it believes."""
        return self._read(self.link.proprioception)

    def imaging(self) -> bool:
        """Whether the world is delivering RGB-D (phase 1) rather than labelled detections now."""
        return self.sensors().get("imaging") == "rgbd"

    def rgbd_frame(self) -> dict:
        return self._read(lambda: decode_rgbd(self.link.rgbd(camera=SURVEY_CAMERA)), "header")

    def _read(self, fetch: Callable[[], dict], header: str | None = None) -> dict:
        """One reading of the robot, clocked. Reads are serialised: the mission, the telemetry
        loop and the Ego-State view all read, and two overlapping requests could answer out of
        order -- which would look exactly like the clock going backwards."""
        with self._clock_lock:
            frame = fetch()
            fields = frame[header] if header else frame
            self._clock(fields.get("stamp"), fields.get("episode"))
            return frame

    def frame_source(self) -> tuple[Callable[[], dict], bool]:
        """``(frame function, rgbd flag)`` for the kind of sensor data the world delivers now."""
        rgbd = self.imaging()
        return (self.rgbd_frame if rgbd else self.sensors), rgbd

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="cobot-ego", daemon=True)
        self._telemetry = threading.Thread(target=self._telemetry_loop, name="cobot-telemetry", daemon=True)
        self._thread.start()
        self._telemetry.start()

    def stop(self) -> None:
        self._running = False
        self._reset.set()  # release a wait for the next episode
        try:
            self.skills.cancel()
        except Exception:  # noqa: BLE001 - stopping must not depend on the skill layer's state
            pass
        for thread in (self._thread, self._telemetry):
            if thread is not None:
                thread.join(timeout=5.0)

    def _loop(self) -> None:
        while self._running:
            try:
                if self.task is None:
                    self._idle_look()
                else:
                    self._round()
            except Exception as exc:  # noqa: BLE001 - reported; the runtime stays up like a robot would
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.report({"cobotEgo": "error", "error": self.last_error})
                time.sleep(1.0)

    def _round(self) -> None:
        self._reset.clear()
        frame, rgbd = self.frame_source()
        try:
            result = MISSIONS[self.task](self.estimator, self.skills, frame, rgbd=rgbd,
                                         **self.mission_kwargs)
        except Exception as exc:  # noqa: BLE001 - a crashed round is a failed round
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "log": []}
        self._settle(result)
        self.rounds += 1
        self.results.append(result)
        self.last_error = None if result.get("ok") else (result.get("error") or "the mission did not complete")
        self.report({"mission": self.task, "round": self.rounds, **result})
        self._await_new_episode()
        if self._running:
            self.estimator.reboot()
            self.report({"cobotEgo": "reboot", "reason": "the world's clock went back: new episode"})

    def _settle(self, result: dict) -> None:
        """Look once more where the round ended, so the belief says what is true now.

        A round's last act is usually a release, and the belief only drops a held track when a
        perception frame arrives after the gripper opened. Without a look here the round parks
        with the load still believed in the hand -- and the Ego State shows the bowl riding up
        with the retreating arm while it lies on the counter. This is also what a real operator
        would ask for: look at what you did before you say you are done."""
        frame, rgbd = self.frame_source()
        try:
            perceive(self.estimator, frame, frames=6 if rgbd else 4, rgbd=rgbd, pause_s=0.0)
        except Exception as exc:  # noqa: BLE001 - a failed look must not bury the round's result
            result.setdefault("log", []).append(f"settling look failed: {type(exc).__name__}: {exc}")

    def _idle_look(self) -> None:
        """No mission: keep looking so the Ego-State view shows what the twin believes."""
        if self._reset.is_set():
            self._reset.clear()
            self.estimator.reboot()
        frame, rgbd = self.frame_source()
        perceive(self.estimator, frame, frames=4, rgbd=rgbd)
        time.sleep(IDLE_LOOK_S)

    def _await_new_episode(self) -> None:
        """Hold the parked pose until the world starts over."""
        while self._running and not self._reset.is_set():
            self.proprioception()
            self._reset.wait(RESET_POLL_S)

    def _telemetry_loop(self) -> None:
        """Telemetry to the Observer at a steady rate, during rounds as well as between them: the
        belief with the body as the encoders report it now. One-way; a lost sample is no event."""
        while self._running:
            try:
                self.link.telemetry(self.estimator.to_scene(self.proprioception()).to_dict())
            except Exception as exc:  # noqa: BLE001 - see above
                self.report({"cobotEgo": "telemetry", "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(TELEMETRY_S)


def _print_line(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False), flush=True, file=sys.stdout)
