"""Steward host: WSim, Observer and authoring behind the World Control API, plus robot ports.

Two kinds of listeners keep the information boundary explicit:

* the **World Control API** (WCA) port serves World State, clock and episode control, rendering,
  Observer reports and asset/scene/scenario authoring. Only the Steward plane (the World
  Harness, operators, tests) may reach it;
* one **robot port** per robot serves only that robot's sensor frames, camera images, actuator
  commands and one-way telemetry. Nothing on a robot port returns World State identifiers.

The console's World State view renders on the host's own view loop thread
(:class:`~steward.world.world_view.WorldViewRenderer`); the Ego-State view renders on the robot,
in the Ego Runtime, and the host only relays its frames. Both follow the same view settings and
the shared frame clock of :mod:`steward.viewloop`.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from steward.authoring.service import AuthoringError, AuthoringService
from steward.embodiments import EMBODIMENTS, embodiment_catalog, scenario_embodiments, with_body
from steward.glrender import IMAGE_FORMATS
from steward.net import NO_CONTENT, Binary, HttpError, JsonServer, call, fetch
from steward.observer.observer import Observer
from steward.world.console import CONSOLE_HTML, console_state, console_thinking_text
from steward.phases import DEFAULT_PHASE, phase_catalog, phase_modules
from steward.scene.extent import entity_body_ids
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState
from steward.world.chat import ChatConfig, HarnessChat
from steward.viewloop import ViewLoop, ViewSettings, frame_response
from steward.world.goals import evaluate as evaluate_goal
from steward.world.sim import EditError, PerceptionConfig, WorldSim, speed_json
from steward.world.sensor_cameras import SensorCameras
from steward.world.thinking import session_files
from steward.world.world_view import WorldViewRenderer


def image_format(query: dict) -> str:
    """``format`` query parameter of an image route: ``png`` (default) or ``jpeg``."""
    fmt = query.get("format", "png")
    if fmt not in IMAGE_FORMATS:
        raise HttpError(400, f"format must be one of {sorted(IMAGE_FORMATS)}")
    return fmt


@dataclass
class _MainTask:
    fn: Any
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


def start_blocked(scenario: dict, tolerance: float = 0.01) -> list[str]:
    """What a robot of ``scenario`` interpenetrates by more than ``tolerance`` at the start, from
    the compiled scene's contacts: another body may be larger than the one the scene was built with."""
    build = build_mjcf(SceneState.from_dict(scenario["sceneState"]), timestep=0.005)
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    mujoco.mj_forward(model, data)
    owner = {bid: eid for eid, names in build.names.items() for bid in entity_body_ids(model, names)}
    robots, out = set(scenario["robots"]), []
    for c in range(data.ncon):
        con = data.contact[c]
        e1, e2 = owner.get(int(model.geom_bodyid[con.geom1])), owner.get(int(model.geom_bodyid[con.geom2]))
        if e1 != e2 and (e1 in robots or e2 in robots) and con.dist < -tolerance:
            out.append(f"{e1} and {e2} interpenetrate by {-con.dist:.3f} m")
    return sorted(set(out))


class StewardHost:
    def __init__(self, root: Path | str, *, scenario: str | None = None, speed: float = 1.0, seed: int = 0,
                 run_dir: Path | None = None, wca_port: int = 0, robot_ports: dict[str, int] | None = None,
                 chat: ChatConfig | None = None, phase: str | None = None, fallback_phase: str = DEFAULT_PHASE,
                 bind_wait_s: float = 0.0, pause_on_perception_fault: bool = False, freeze_idle: bool = False,
                 appearance: str = "flat", misnames: dict[str, str] | None = None, body: str | None = None,
                 dispatch_task: bool = False):
        # ``phase`` pins the level for every episode; None: each scenario's declared level, or
        # ``fallback_phase`` for a scenario that declares none.
        phase_modules(phase or fallback_phase)
        # Names the ideal detections report instead of the true categories at a ``misnamed`` level.
        self.misnames = dict(misnames or {})
        # The body every episode's robots get; None: the body each scenario was built with.
        if body is not None and body not in EMBODIMENTS:
            raise ValueError(f"unknown body {body!r}; registered: {', '.join(EMBODIMENTS)}")
        self.body = body
        self._pinned_phase = phase
        self._fallback_phase = fallback_phase
        self.phase = phase or fallback_phase
        self.authoring = AuthoringService(root)
        self.chat = HarnessChat(chat, lambda: self.wca.url)
        # Session files of reset episodes, whose thinking the console no longer shows.
        self.stale_sessions: frozenset[str] = frozenset()
        # On: every episode the WCA loads or resets starts the robot's Ego Harness on the scenario's
        # task (:meth:`dispatch_scenario_task`), and a turn still running is stopped before the world changes.
        self.dispatch_task = dispatch_task
        self.speed = speed
        self.freeze_idle = freeze_idle
        self.appearance = appearance
        self.seed = seed
        self.run_dir = run_dir
        self.sim: WorldSim | None = None
        self.sensor_cameras = SensorCameras(self._require_sim)
        self.scenario: dict | None = None
        self.embodiments: dict[str, dict] = {}
        self.observer = Observer(self._require_sim, run_dir, phase=lambda: self.phase, on_anomaly=self._perception_fault)
        self.pause_on_perception_fault = pause_on_perception_fault
        self.perception_alerts: list[dict] = []
        self._paused_by_monitor = False
        self._alert_lock = threading.Lock()
        self.bind_wait_s = bind_wait_s
        self.wca = JsonServer(self._wca_routes(), port=wca_port, name="wca", bind_wait_s=bind_wait_s)
        self._robot_port_config = robot_ports or {}
        self.robot_servers: dict[str, JsonServer] = {}
        self._main_tasks: queue.Queue[_MainTask] = queue.Queue()
        self.ego_apis: dict[str, str] = {}
        # Called with the scenario about to load, before the world it replaces stops, and again
        # after it is up: `steward up` stops the Ego Runtimes of bodies that are leaving and
        # starts the ones the new bodies need. Stopping first matters -- a runtime whose clock
        # went backwards would otherwise begin a round against a scenario it is not built for.
        self.before_episode_load: Callable[[dict], None] | None = None
        self.on_episode_loaded: Callable[[], None] | None = None
        # Called on a reset, before the world starts over: `steward up` stops every Ego Runtime
        # process, so that :attr:`on_episode_loaded` starts each robot afresh, with nothing it believed,
        # planned or recorded left over.
        self.before_episode_reset: Callable[[], None] | None = None
        self.view_settings: ViewSettings | None = None
        renderer = WorldViewRenderer(lambda: self.sim)
        self.world_view = ViewLoop("world", renderer.render, renderer.close)
        if scenario is not None:
            self.load_episode(scenario)

    # --------------------------------------------------------------- episodes
    def scenario_for(self, name: str) -> dict:
        """Scenario ``name`` as it will load: built on first use, its robots given :attr:`body`."""
        # A builtin scenario the operator picked from the catalog is built on first use, so the
        # console offers every scene without a separate build command per root.
        self.authoring.ensure_scenario(name)
        scenario = self.authoring.load_scenario(name)
        if self.body is None or set(scenario_embodiments(scenario)[r]["id"] for r in scenario["robots"]) == {self.body}:
            return scenario
        bodied = with_body(scenario, self.body)
        blocked = start_blocked(bodied)
        if blocked:
            raise HttpError(409, f"{EMBODIMENTS[self.body]['label']} does not fit where scenario {name!r} starts its "
                                 f"robot: {'; '.join(blocked)}")
        return bodied

    def load_episode(self, name: str) -> dict:
        scenario = self.scenario_for(name)
        if scenario.get("phase"):
            phase_modules(scenario["phase"])  # a scenario declares the level it was built for; unknown ones fail here
        self.phase = self._pinned_phase or scenario.get("phase") or self._fallback_phase
        self.sensor_cameras.recordings.interrupt("Scene changed; start a new RGB recording")
        if self.sim is not None:
            self.sim.stop()
        self.sim = WorldSim(scenario["sceneState"], speed=self.speed, seed=self.seed, freeze_idle=self.freeze_idle,
                            appearance=self.appearance,
                            perception=PerceptionConfig(mode=phase_modules(self.phase)["perception"],
                                                        masks=phase_modules(self.phase).get("masks", True),
                                                        misnames=self._misnames_at(self.phase)))
        self.scenario = scenario
        self.embodiments = scenario_embodiments(scenario)
        self.observer.reset()
        # The chat and the console name the robot the scenario actually has, not a fixed one.
        robot_label = next(iter(scenario["robots"]), "robot-1").removeprefix("w:")
        if "robot" in self.chat.agents:
            self.chat.agents["robot"].label = robot_label
            # Tell the Ego Harness which body it is (single-hand Steward vs dual-arm Cobot).
            body_id = next((e["id"] for e in self.embodiments.values() if e.get("registered")), None)
            self.chat.set_robot_embodiment(body_id)
        # A robot that left with the previous episode takes its port with it: the port is how
        # clients reach a robot, and nothing is behind it once the robot is gone.
        for rid in list(self.robot_servers):
            if rid not in scenario["robots"]:
                self.robot_servers.pop(rid).stop()
                self.detach_ego_api(rid)
        for index, rid in enumerate(scenario["robots"]):
            if rid not in self.robot_servers:
                port = self._robot_port_config.get(rid)
                if port is None:
                    # The configured fixed port goes to the scenario's first robot, whichever body
                    # it is; any further robot of the same scenario takes a free one.
                    port = self._robot_port_config.get("*", 0) if index == 0 else 0
                server = JsonServer(self._robot_routes(rid), port=port, name=f"robot-{rid}",
                                    bind_wait_s=self.bind_wait_s)
                self.robot_servers[rid] = server.start()
        return {"scenario": name, "robots": {rid: self.robot_servers[rid].url for rid in scenario["robots"]},
                "task": scenario["task"]}

    def set_body(self, body: str | None) -> None:
        """The body later episodes give their robots; None: each scenario's own."""
        if body is not None and body not in EMBODIMENTS:
            raise HttpError(400, f"unknown body {body!r}; registered: {', '.join(EMBODIMENTS)}")
        self.body = body

    def set_phase(self, level: str) -> dict:
        """Switch the running system to a phase level (module implementations), effective with the
        next sensor frame. Later episodes start at the selected level.

        If the level or its required name mapping is invalid, the active and pinned levels
        and the running simulation's perception settings are preserved.
        """
        modules = phase_modules(level)
        misnames = self._misnames_at(level)
        self.phase = self._pinned_phase = level
        if self.sim is not None:
            self.sim.perception.mode = modules["perception"]
            self.sim.perception.masks = modules.get("masks", True)
            self.sim.perception.misnames = misnames
            self.sim.events.append({"t": self.sim.time, "type": "phase/switch", "level": level})
        return self.phase_state()

    def _misnames_at(self, level: str) -> dict[str, str]:
        if phase_modules(level).get("misnamed") and not self.misnames:
            raise ValueError(f"phase level {level!r} misnames objects, but this host has no misnames configured")
        return dict(self.misnames) if phase_modules(level).get("misnamed") else {}

    def phase_state(self) -> dict:
        return {"level": self.phase, "levels": phase_catalog()}

    def goal_progress(self) -> dict | None:
        """Progress towards the loaded scenario's ``goal`` from the World State, or None without one."""
        goal = (self.scenario or {}).get("goal")
        if goal is None or self.sim is None:
            return None
        sim = self.sim
        with sim.lock:
            return evaluate_goal(goal, sim._entity_pose)

    def _require_sim(self) -> WorldSim:
        if self.sim is None:
            raise HttpError(409, "no episode is loaded")
        return self.sim

    def on_main(self, fn, timeout: float = 120.0):
        """Run ``fn`` on the process main thread and return its result.

        Anything that creates, uses or releases OpenGL renderers (episode load and reset,
        previews) must run on the main thread on macOS; request threads queue it here and the
        main thread executes it in :meth:`pump`.
        """
        if threading.current_thread() is threading.main_thread():
            return fn()
        task = _MainTask(fn)
        self._main_tasks.put(task)
        if not task.done.wait(timeout):
            raise HttpError(503, "the Steward main thread did not run the request")
        if task.error is not None:
            raise task.error
        return task.result

    def pump(self, timeout: float = 0.05) -> None:
        """Run queued main-thread tasks and serve render requests; call repeatedly from the main thread."""
        while True:
            try:
                task = self._main_tasks.get_nowait()
            except queue.Empty:
                break
            try:
                task.result = task.fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the requesting thread
                task.error = exc
            task.done.set()
        if self.sim is not None:
            self.sim.pump(timeout)
        else:
            time.sleep(timeout)

    def attach_ego_api(self, robot: str, url: str, skills: bool = True) -> None:
        """Relay the console's Ego-State view of ``robot`` from the Ego API at ``url``.

        ``skills`` says whether that Ego API also takes work (``/skill/*``); a body whose runtime
        only publishes its belief is shown in the console, but an Ego Harness task addressed to it
        is refused up front instead of failing on every tool call."""
        self.ego_apis[robot] = url
        self.chat.robot_reason = None if skills else (
            f"this robot's body publishes its Ego-State but takes no work: its Ego API has no "
            f"skill endpoints, so an Ego Harness task has nothing to call")
        if self.view_settings is not None:
            self._configure_robot_view(url)

    def detach_ego_api(self, robot: str) -> None:
        """Stop relaying ``robot``'s Ego-State view (its Ego Runtime is gone or being replaced)."""
        self.ego_apis.pop(robot, None)

    def dispatch_scenario_task(self) -> None:
        """With :attr:`dispatch_task` on, start the robot's Ego Harness on the loaded scenario's task
        in a new conversation; what keeps it from starting is reported in the chat."""
        if not self.dispatch_task or "robot" not in self.chat.agents or self.scenario is None:
            return
        label = self.chat.agents["robot"].label
        task = self.scenario["task"].strip()
        if not task:
            self.chat.post_note("robot", f"场景 {self.scenario['name']} 没有任务，没有下发给 {label}")
            return
        try:
            self.chat.assign("robot", task)
        except HttpError as exc:
            self.chat.post_error("robot", f"未能把场景任务下发给 {label}：{exc}")

    def _start_robot_runtimes(self) -> None:
        """Run :attr:`on_episode_loaded` once the episode's world is up; raises ``HttpError(500)``
        when the robots' runtimes did not start."""
        if self.on_episode_loaded is None:
            return
        try:
            self.on_episode_loaded()
        except Exception as exc:  # noqa: BLE001 - the world is up; say what is not
            raise HttpError(500, f"the episode is loaded but its robot runtime is not: "
                                 f"{type(exc).__name__}: {exc}") from exc

    def clear_transcripts(self) -> None:
        """Empty the chat transcripts and the robot's thinking of the episode being reset (the
        sessions that exist now are no longer read); its plan and activity record go with its Ego
        Runtime process (:attr:`before_episode_reset`)."""
        self.chat.clear()
        home = getattr(self.chat.config, "robot_home", None)
        if home is not None:
            self.stale_sessions = frozenset(session_files(str(home)))

    def _stop_robot_turn(self) -> None:
        """With :attr:`dispatch_task` on, end the robot's running turn before the world it acts in changes."""
        if self.dispatch_task and "robot" in self.chat.agents:
            self.chat.stop_turn("robot")

    def start(self) -> StewardHost:
        self.wca.start()
        self.world_view.start()
        if self.sim is not None:
            self.sim.start()
        return self

    def stop(self) -> None:
        self.chat.stop()
        for server in self.robot_servers.values():
            server.stop()
        self.wca.stop()
        self.world_view.stop()
        self.sensor_cameras.close()
        if self.sim is not None:
            self.sim.stop()

    # ------------------------------------------------------------------- WCA
    def _wca_routes(self) -> dict:
        a = self.authoring

        def guard(fn):
            def wrapped(q, b):
                try:
                    return fn(q, b or {})
                except AuthoringError as exc:
                    raise HttpError(400, str(exc)) from None
                except KeyError as exc:
                    raise HttpError(404, str(exc)) from None
            return wrapped

        def episode(q, b):
            action = b.get("action")
            if action == "load":
                if "body" in b:
                    self.set_body(b["body"] or None)
                self._stop_robot_turn()

                def load() -> dict:
                    if self.before_episode_load is not None:
                        self.before_episode_load(self.scenario_for(b["scenario"]))
                    result = self.load_episode(b["scenario"])
                    self.sim.start()
                    return result
                result = self.on_main(load)
                self._start_robot_runtimes()
                self.dispatch_scenario_task()
                return result
            if action == "reset":
                sim = self._require_sim()
                if "robot" in self.chat.agents:
                    self.chat.stop_turn("robot")
                self.clear_transcripts()
                if self.before_episode_reset is not None:
                    self.before_episode_reset()
                self.sensor_cameras.recordings.interrupt("Scene reset; start a new RGB recording")
                self.on_main(sim.reset)
                self.observer.reset()
                self._start_robot_runtimes()
                self.dispatch_scenario_task()
                return {"reset": True}
            raise HttpError(400, "action must be load or reset")

        def clock(q, b):
            sim = self._require_sim()
            action = b.get("action")
            if action == "pause":
                sim.paused = True
            elif action == "resume":
                sim.paused = False
            elif action == "speed":
                sim.speed = math.inf if b["value"] == "max" else float(b["value"])
            else:
                raise HttpError(400, "action must be pause, resume or speed")
            return {"paused": sim.paused, "speed": speed_json(sim.speed), "t": sim.time}

        def render(q, b):
            sim = self._require_sim()
            camera = q.get("camera", "overview")
            if camera.startswith("robot:"):
                try:
                    camera = sim.robot_camera_name(camera.removeprefix("robot:"), q.get("sensor"))
                except KeyError as exc:
                    raise HttpError(404, f"no such robot camera: {exc}") from None
            elif camera == "free":
                try:
                    lookat = [float(v) for v in q["lookat"].split(",")]
                    camera = {"lookat": lookat, "distance": float(q["distance"]), "azimuth": float(q["azimuth"]),
                              "elevation": float(q["elevation"])}
                except (KeyError, ValueError) as exc:
                    raise HttpError(400, f"free camera needs lookat=x,y,z, distance, azimuth, elevation: {exc}") from None
                if len(lookat) != 3 or camera["distance"] <= 0:
                    raise HttpError(400, "free camera needs three lookat values and a positive distance")
            fmt = image_format(q)
            return Binary(sim.render(camera, int(q.get("width", 800)), int(q.get("height", 600)), fmt=fmt),
                          IMAGE_FORMATS[fmt])

        def console_page(q, b):
            return Binary(CONSOLE_HTML.read_bytes(), "text/html; charset=utf-8")

        def frame(q, b):
            after = q.get("after")
            if q.get("view", "world") == "world":
                found = self.world_view.latest(int(after) if after else None)
                if found is not None:
                    return frame_response(found, self.world_view.status())
                status = self.world_view.status()
                if status["error"] and status["frame"] is None:
                    raise HttpError(503, f"the world view cannot render: {status['error']}")
                return NO_CONTENT
            return self._relay_ego_frame(q.get("robot") or next(iter(self.ego_apis), None), after)

        return {
            ("GET", "/"): console_page,
            ("GET", "/console"): console_page,
            ("GET", "/console/state"): guard(lambda q, b: console_state(self)),
            ("GET", "/console/thinking"): guard(lambda q, b: console_thinking_text(self, q.get("robot", ""), q.get("entry", ""))),
            ("GET", "/console/frame"): guard(frame),
            ("GET", "/console/view"): guard(lambda q, b: self.view_state()),
            ("POST", "/console/view"): guard(lambda q, b: self.set_view(b)),
            ("GET", "/wca/status"): guard(lambda q, b: {"scenario": (self.scenario or {}).get("name"),
                                                        **(self.sim.status() if self.sim else {"loaded": False})}),
            ("GET", "/wca/scene"): guard(lambda q, b: self._require_sim().world_state()),
            ("GET", "/wca/goal"): guard(lambda q, b: {"goal": self.goal_progress()}),
            ("POST", "/wca/clock"): guard(clock),
            ("POST", "/wca/episode"): guard(episode),
            ("GET", "/wca/render"): guard(render),
            ("GET", "/observer/latest"): guard(lambda q, b: self.observer.latest),
            ("GET", "/observer/report"): guard(lambda q, b: self.observer.report()),
            ("GET", "/assets/generators"): guard(lambda q, b: {"generators": a.generators()}),
            ("GET", "/assets"): guard(lambda q, b: {"assets": a.library.search(q.get("query"), q.get("category"))}),
            ("GET", "/assets/get"): guard(lambda q, b: a.library.get(q["name"], q.get("version"))),
            ("POST", "/assets/create"): guard(lambda q, b: a.create_asset(b.get("generator"), b.get("params"),
                                                                          b.get("template"))),
            ("POST", "/assets/preview"): guard(lambda q, b: {"images": self.on_main(
                lambda: a.preview_asset(b["template"], b.get("stem", "draft")))}),
            ("POST", "/assets/publish"): guard(lambda q, b: self.on_main(
                lambda: a.publish_asset(b["name"], b["template"], b.get("meta")))),
            ("GET", "/scenes"): guard(lambda q, b: {"scenes": a.list_scenes()}),
            ("POST", "/scenes/compose"): guard(lambda q, b: a.compose_scene(b)),
            ("POST", "/scenes/render"): guard(lambda q, b: {"images": self.on_main(
                lambda: a.render_scene_views(b["name"]))}),
            ("GET", "/scenarios"): guard(lambda q, b: {"scenarios": a.list_scenarios(),
                                                       "catalog": a.scenario_catalog(),
                                                       "bodies": embodiment_catalog(), "body": self.body}),
            ("POST", "/scenarios/save"): guard(lambda q, b: a.save_scenario(b)),
            ("POST", "/wca/edit"): guard(lambda q, b: self.edit(b)),
            ("GET", "/wca/phase"): guard(lambda q, b: self.phase_state()),
            ("POST", "/wca/phase"): guard(lambda q, b: self._set_phase_request(b)),
            ("GET", "/console/chat"): guard(lambda q, b: self.chat.state(int(q.get("since", 0)), int(q.get("activity", 0)))),
            ("POST", "/console/chat"): guard(lambda q, b: self.chat.send(b.get("agent", "world"), b.get("message", ""))),
            ("POST", "/console/chat/reset"): guard(lambda q, b: self.chat.reset(b.get("agent", "world"))),
            ("POST", "/console/chat/stop"): guard(lambda q, b: self.chat.cancel(b.get("agent", "world"))),
            ("POST", "/console/alerts/ack"): guard(lambda q, b: self.acknowledge_alerts()),
        }

    # ------------------------------------------------- perception fault monitor
    def _perception_fault(self, robot: str, anomaly: dict) -> None:
        """A gross perception fault the Observer found: alert the operator and, when configured,
        pause the world so the robot does not act on a badly wrong Ego-State."""
        sim = self.sim
        with self._alert_lock:
            record = {"id": len(self.perception_alerts) + 1, "t": round(sim.time, 2) if sim else 0.0, "robot": robot,
                      "kind": anomaly["kind"], "message": anomaly["message"], "analysis": anomaly["analysis"],
                      "acknowledged": False}
            self.perception_alerts.append(record)
        paused = " 仿真已暂停，点「解除告警」后继续。" if self.pause_on_perception_fault else ""
        self.chat.post_error("robot", f"⚠ 世界侧感知监测：{anomaly['message']}。{anomaly['analysis']}{paused}")
        if sim is not None:
            sim.events.append({"t": sim.time, "type": "observer/perception-fault", "robot": robot, "kind": anomaly["kind"],
                               "entity": anomaly["entity"]})
            if self.pause_on_perception_fault and not sim.paused:
                sim.paused = True
                self._paused_by_monitor = True

    def active_perception_alerts(self) -> list[dict]:
        with self._alert_lock:
            return [dict(a) for a in self.perception_alerts if not a["acknowledged"]]

    def acknowledge_alerts(self) -> dict:
        """Acknowledge robot alerts and perception-fault alerts; a world paused by the monitor resumes."""
        with self._alert_lock:
            for record in self.perception_alerts:
                record["acknowledged"] = True
        if self._paused_by_monitor and self.sim is not None:
            self.sim.paused = False
            self._paused_by_monitor = False
        robot = self.chat.acknowledge_alerts() if self.chat.config and self.chat.config.ego_api else {}
        return {"robot": robot, "perception": self.active_perception_alerts()}

    # ------------------------------------------------------------------ views
    def set_view(self, body: dict) -> dict:
        """Apply console view settings to the World State view and every attached robot's Ego-State
        view, so both render the same camera at the same rate on the shared frame clock."""
        try:
            self.view_settings = ViewSettings.parse(body)
        except ValueError as exc:
            raise HttpError(400, str(exc)) from None
        self.world_view.configure(self.view_settings)
        for url in self.ego_apis.values():
            self._configure_robot_view(url)
        return self.view_state()

    def view_state(self) -> dict:
        """Status of the World State view and of each robot's Ego-State view (``error`` when unreachable)."""
        robots = {}
        for robot, url in self.ego_apis.items():
            try:
                robots[robot] = call("GET", f"{url}/ego/view", timeout=1.0)
            except (HttpError, OSError) as exc:  # the robot's view is reported as unavailable
                robots[robot] = {"error": f"Ego API unreachable: {exc}"}
        return {"world": self.world_view.status(), "ego": robots}

    def _configure_robot_view(self, url: str) -> None:
        try:
            call("POST", f"{url}/ego/view", self.view_settings.to_dict(), timeout=1.0)
        except (HttpError, OSError):  # the next frame request retries; view_state reports the failure
            pass

    def _relay_ego_frame(self, robot: str | None, after: str | None):
        """The newest Ego-State frame from the robot's Ego API, passed through unchanged."""
        url = self.ego_apis.get(robot or "")
        if url is None:
            raise HttpError(404, f"no Ego API is attached for robot {robot!r}")
        try:
            status, body, headers = fetch(f"{url}/ego/view/frame" + (f"?after={int(after)}" if after else ""), timeout=2.0)
        except HttpError as exc:
            if exc.status == 409:
                # The robot's view has no settings yet (it just started, or the console has not sent
                # its camera yet): pass ours on if we have them; either way a frame follows shortly.
                if self.view_settings is not None:
                    self._configure_robot_view(url)
                return NO_CONTENT
            raise
        except OSError as exc:
            raise HttpError(502, f"the Ego API of {robot} is unreachable: {exc}") from None
        if status == 204:
            return NO_CONTENT
        relayed = {k: v for k, v in headers.items() if k.startswith("X-") or k == "Cache-Control"}
        return Binary(body, headers.get("Content-Type", "image/jpeg"), relayed)

    def _set_phase_request(self, body: dict) -> dict:
        try:
            return self.set_phase(str(body.get("level", "")))
        except ValueError as exc:
            raise HttpError(400, str(exc)) from None

    # -------------------------------------------------------------- edits
    def edit(self, request: dict) -> dict:
        """Apply one live World State edit (``move``, ``spawn`` or ``remove``) on the main thread.

        Placement fields are ``at`` {x, y, yaw?, z?} or ``on`` (support id) with ``offset``
        [dx, dy] and ``yaw``. Entity ids may omit the ``w:`` prefix."""
        sim = self._require_sim()
        action = request.get("action")
        entity = request.get("entity")
        if not entity:
            raise HttpError(400, "edit needs an entity id")
        entity = self._world_id(sim, entity, must_exist=action != "spawn")
        placement = {k: request[k] for k in ("at", "offset", "yaw") if request.get(k) is not None}
        if request.get("on"):
            placement["on"] = self._world_id(sim, request["on"], must_exist=True)
        try:
            if action == "move":
                result = self.on_main(lambda: sim.edit_move(entity, placement))
            elif action == "spawn":
                if not request.get("asset"):
                    raise HttpError(400, "spawn needs an asset name")
                record = self.authoring.library.get(request["asset"], request.get("version"))
                asset = {"id": request["asset"], "version": record["version"]}
                result = self.on_main(lambda: sim.edit_spawn(entity, record["template"], placement, asset))
            elif action == "remove":
                result = self.on_main(lambda: sim.edit_remove(entity))
            else:
                raise HttpError(400, "action must be move, spawn or remove")
        except EditError as exc:
            raise HttpError(409, str(exc)) from None
        return result

    @staticmethod
    def _world_id(sim: WorldSim, entity: str, *, must_exist: bool) -> str:
        if entity.startswith("w:"):
            return entity
        prefixed = f"w:{entity}"
        if not must_exist or sim.template.find(prefixed) is not None:
            return prefixed
        return entity

    # ------------------------------------------------------------ robot port
    def _robot_routes(self, rid: str) -> dict:
        def camera_id(q) -> str | None:
            """The ``camera`` query (a sensor id of this robot; absent: its primary camera)."""
            camera = q.get("camera")
            if camera is not None and camera not in {c["id"] for c in self._require_sim().robot_cameras(rid)}:
                raise HttpError(404, f"robot {rid} has no camera {camera!r}")
            return camera

        def camera(q, b):
            sim = self._require_sim()
            fmt = image_format(q)
            image = sim.render(sim.robot_camera_name(rid, camera_id(q)), int(q.get("width", 320)),
                               int(q.get("height", 240)), fmt=fmt)
            return Binary(image, IMAGE_FORMATS[fmt])

        def clock_wait(q, b):
            """Wait for the world clock (:meth:`WorldSim.wait_until`): ``until`` (or ``seconds`` from now;
            neither: now), ``hold``, ``rank`` and ``pulls``. Answers 504 after ``timeoutS`` wall seconds (default 5)
            without waking; the caller waits again. A timeout while the world is paused says so: blocking is the
            right answer to an operator pause, but the caller deserves the reason rather than a bare step number."""
            sim = self._require_sim()
            until = float(b["until"]) if "until" in b else None
            hold = f"{rid}/{b['hold']}" if b.get("hold") else None
            try:
                return {"t": sim.wait_until(until, hold, timeout=float(b.get("timeoutS", 5.0)),
                                            rank=int(b.get("rank", 0)), seconds=float(b.get("seconds", 0.0)),
                                            pulls=bool(b.get("pulls", True)))}
            except TimeoutError as exc:
                message = str(exc)
                if sim.paused:
                    message += ("; the world is paused" + (
                        " by a perception fault — acknowledge it (POST /console/alerts/ack)" if self._paused_by_monitor
                        else "") + ' — resume it (POST /wca/clock {"action": "resume"}) to continue')
                raise HttpError(504, message) from exc

        def telemetry(q, b):
            if not isinstance(b, dict):
                raise HttpError(400, "telemetry body must be an ego SceneState")
            self.observer.ingest(rid, b)
            return {"accepted": True}

        return {
            ("GET", "/sensors"): lambda q, b: self._require_sim().sensors(rid),
            # The robot's own encoders without a perception sample: an Ego-State view polling
            # them at frame rate must not advance the detector's noise stream.
            ("GET", "/sensors/proprioception"): lambda q, b: self._require_sim().proprioception(rid),
            ("GET", "/sensors/rgbd"): lambda q, b: Binary(self._require_sim().rgbd(rid, camera_id(q)), "application/octet-stream"),
            ("GET", "/sensors/cameras"): lambda q, b: self.sensor_cameras.sensor_snapshot(rid),
            ("GET", "/sensors/kinematics"): lambda q, b: self.sensor_cameras.kinematics(rid),
            ("GET", "/sensors/cameras/calibration"): lambda q, b: self.sensor_cameras.calibration(rid),
            ("POST", "/sensors/rgb/record/start"): lambda q, b: self.sensor_cameras.recordings.start(rid, b or {}),
            ("GET", "/sensors/rgb/record/status"): lambda q, b: self.sensor_cameras.recordings.get(rid, q["recording_id"]).status(),
            ("POST", "/sensors/rgb/record/stop"): lambda q, b: self.sensor_cameras.recordings.get(rid, b["recording_id"]).finish(b.get("cancel", False)),
            ("GET", "/sensors/rgb/record/result"): lambda q, b: self.sensor_cameras.recordings.get(rid, q["recording_id"]).result(),
            ("GET", "/sensors/rgb/pose"): lambda q, b: self.sensor_cameras.odometry_kinematics(rid),
            ("GET", "/camera/rgb"): camera,
            ("POST", "/command"): lambda q, b: self._require_sim().command(rid, b or {}),
            ("POST", "/telemetry"): telemetry,
            ("POST", "/clock/wait"): clock_wait,
            ("POST", "/clock/release"): lambda q, b: self._require_sim().release(f"{rid}/{b['hold']}") or {},
            ("POST", "/clock/renew"): lambda q, b: self._require_sim().renew(f"{rid}/{b['hold']}") or {},
        }
