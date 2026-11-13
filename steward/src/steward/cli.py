"""Steward command line.

    steward build-kitchen            generate, validate and publish the Phase 0 kitchen
    steward build-toy-assembly       two tables of disassembled airplane and car toys with screws
    steward build-screw-board        table with a four-hole board, four screws, and a screwdriver
    steward world                    run WSim + Observer + WCA + robot ports
    steward ego                      run the mobile robot's Ego Runtime + Ego API
    steward cobot-ego                run the Cobot Magic twin's Ego Runtime + Ego API (+ its scripted mission)
    steward up                       run the world, and each robot's Ego Runtime in a process of its body's kind
    steward demo                     scripted "cup to table" mission, checked against World State
    steward shuttle                  carry objects around table → cabinet → sideboard, step by step
    steward chores                   sort tableware, stack it on the sideboard, swap drawer contents; repeat
    steward cobot                    the twin with its console in one process, rounds driven by world edits
    steward services                 list the remote perception services and check that each is reachable
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from steward.embodiments import EMBODIMENTS
from steward.phases import LATEST_PHASE, PHASES, load_misnames
from steward.world.chat import TURN_BUDGET_S

STEWARD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = STEWARD_DIR / "var" / "steward"
SEGMENTATION_CONFIG = STEWARD_DIR / "config" / "segmentation.toml"
MISNAMES_CONFIG = STEWARD_DIR / "config" / "misnames.toml"

# The Cobot Magic twin keeps its own ports: the kitchen scenario owns 7400/7410, and two worlds
# that fight over one port is the most confusing failure there is -- the console comes up, but it
# is somebody else's world.
COBOT_WCA_PORT = 7430
COBOT_ROBOT_PORT = 7440
# The twin's scripted missions (steward.ego.cobot_runtime.MISSIONS), named here so parsing the
# command line does not import the skill layer and its IK solver.
COBOT_MISSIONS = ("bowl-on-plate", "nest-bowls", "carry-to-counter")


def _tableware(ent) -> bool:
    """A movable object on the scene: everything an object that is not itself a support surface."""
    if ent.role != "object":
        return False
    return not any(a.value == "support-surface" for link in ent.rigid.links for a in link.affordances)


def _wait_forever(stoppers: list, pump=None) -> None:
    """Block until SIGINT/SIGTERM; the main thread serves render requests through ``pump``."""
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    while not done.is_set():
        if pump is not None:
            pump(0.05)
        else:
            done.wait(0.5)
    for stop in stoppers:
        stop()


def cmd_build_kitchen(args) -> int:
    from steward.authoring.service import AuthoringService, build_default_kitchen

    out = build_default_kitchen(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_build_toy_assembly(args) -> int:
    from steward.authoring.service import AuthoringService, build_toy_assembly

    out = build_toy_assembly(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_build_screw_board(args) -> int:
    from steward.authoring.service import AuthoringService, build_screw_board

    out = build_screw_board(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _host(args, run_dir: Path, start: bool = True):
    from steward.world.server import StewardHost

    _require_free({"console (world)": getattr(args, "wca_port", 0),
                   "robot": getattr(args, "robot_port", 0),
                   "ego api": getattr(args, "api_port", 0)})
    # ``"*"`` is the port the scenario's first robot gets, whatever its id: which body a scenario
    # puts in the world is its business, and the operator asked for a fixed robot port.
    ports = {"*": args.robot_port} if getattr(args, "robot_port", 0) else {}
    host = StewardHost(args.root, scenario=args.scenario, speed=args.speed, seed=args.seed, run_dir=run_dir,
                       wca_port=args.wca_port, robot_ports=ports, chat=_chat_config(args),
                       phase=args.phase, fallback_phase=LATEST_PHASE, bind_wait_s=getattr(args, "bind_wait", 0.0),
                       pause_on_perception_fault=getattr(args, "pause_on_perception_fault", False),
                       freeze_idle=args.freeze_idle, body=getattr(args, "body", None),
                       dispatch_task=getattr(args, "dispatch_task", False),
                       misnames=load_misnames(args.misnames_config) if getattr(args, "misnames_config", "") else None)
    return host.start() if start else host


def _require_free(ports: dict[str, int]) -> None:
    """Fail before anything starts, naming the port, when another steward run still holds it.

    Without this the bind error arrives as a bare "Address already in use" from a background
    thread, and the confusing part is that a console does open -- it is just the other run's."""
    import socket

    for what, port in ports.items():
        if not port:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # A port a previous run left in TIME_WAIT is free for us; only a live listener is not.
            # Windows rebinds TIME_WAIT ports by default, and SO_REUSEADDR there would bind over a live listener.
            if os.name != "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                find = (f"netstat -ano | findstr :{port}" if os.name == "nt"
                        else f"lsof -nP -iTCP:{port} -sTCP:LISTEN")
                raise SystemExit(
                    f"steward: the {what} port {port} is already in use.\n"
                    f"  another steward run is probably still up; find it with\n"
                    f"    {find}\n"
                    f"  then stop it, or pass a different port (--wca-port / --robot-port / --api-port).\n"
                    f"  the kitchen scenario owns 7400/7410/7420; `steward cobot` owns "
                    f"{COBOT_WCA_PORT}/{COBOT_ROBOT_PORT}") from None


def _chat_config(args):
    """Operator chat for the console: the steward-local DSH launcher, the World Harness home and
    the robot's Ego Harness home. ``steward up`` fills in its own Ego API address after start."""
    from steward.world.chat import ChatConfig

    world, robot = getattr(args, "world_harness", None), getattr(args, "robot_harness", None)
    if not world and not robot:
        return None
    _dsh_bin = STEWARD_DIR / "dsh" / "node_modules" / ".bin"
    # Windows cannot exec the POSIX sh launcher; dsh.CMD is the npm-generated cmd wrapper
    _dsh = _dsh_bin / ("dsh.CMD" if os.name == "nt" and (_dsh_bin / "dsh.CMD").exists() else "dsh")
    return ChatConfig(dsh=_dsh, cwd=STEWARD_DIR,
                      env_file=STEWARD_DIR / ".env", world_home=Path(world) if world else None,
                      robot_home=Path(robot) if robot else None, ego_api=getattr(args, "ego_api", None),
                      timeout_s=getattr(args, "turn_budget_s", TURN_BUDGET_S))


def cmd_world(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    host = _host(args, run_dir)
    if args.ego_api:
        # Whatever body this scenario has: its robots are the scenario's, not a fixed id.
        robot = next(iter(host.scenario["robots"]), None)
        if robot is None:
            raise SystemExit("steward world: the scenario has no robots, so there is no robot to attach an Ego API to")
        host.attach_ego_api(robot, args.ego_api,
                            skills=EMBODIMENTS[host.embodiments[robot]["id"]]["skills"] if host.embodiments[robot]["registered"] else False)
    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url, "robots": {k: v.url for k, v in host.robot_servers.items()},
                      "runDir": str(run_dir)}), flush=True)
    _wait_forever([host.stop], host.pump)
    return 0


def _segmenter(config: str, root: str):
    """The robot-side segmenter from ``config`` (``''``: none), caching results under ``<root>/ego/segmentation``."""
    if not config:
        return None
    from steward.ego.segmenter import Segmenter

    return Segmenter.from_files(Path(config), cache_dir=Path(root) / "ego" / "segmentation")


def _start_ego(robot_url: str, api_port: int, out_dir: Path, bind_wait_s: float = 0.0, segmenter=None,
               body: str = "steward-bot"):
    from steward.ego.bodies import body_arm, robot_model, runtime_config
    from steward.ego.runtime import EgoRuntime, RobotLink
    from steward.ego.server import ego_api

    model = robot_model(body)
    runtime = EgoRuntime(RobotLink(robot_url), model, runtime_config(body), out_dir=out_dir, segmenter=segmenter,
                         arm=body_arm(body, model))
    runtime.start()
    api = ego_api(runtime, port=api_port, bind_wait_s=bind_wait_s).start()
    return runtime, api


def cmd_ego(args) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.root) / "ego" / "observations"
    runtime, api = _start_ego(args.robot_url, args.api_port, out_dir, args.bind_wait,
                              _segmenter(args.segmentation_config, args.root), body=args.body)
    print(json.dumps({"egoApi": api.url, "robot": args.robot_url}), flush=True)
    if args.parent_pid:
        _exit_with_parent(args.parent_pid)
    _wait_forever([api.stop, runtime.stop])
    return 0


def cmd_cobot_ego(args) -> int:
    """The Cobot Magic twin's Ego Runtime behind a robot port, as `steward up` runs it."""
    from steward.ego.cobot_runtime import CobotEgoRuntime
    from steward.ego.cobot_server import cobot_ego_api
    from steward.ego.runtime import RobotLink

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.root) / "ego" / "observations"
    mission_kwargs = json.loads(args.mission_config) if args.mission_config else {}
    runtime = CobotEgoRuntime(RobotLink(args.robot_url), task=args.task, out_dir=out_dir,
                              mission_kwargs=mission_kwargs)
    api = cobot_ego_api(runtime.estimator, port=args.api_port, bind_wait_s=args.bind_wait,
                        live=runtime.proprioception)
    # The first line is the contract with the launcher: everything after it is the runtime's own.
    print(json.dumps({"egoApi": api.url, "robot": args.robot_url, "mission": args.task}), flush=True)
    runtime.start()
    if args.parent_pid:
        _exit_with_parent(args.parent_pid)
    _wait_forever([api.stop, runtime.stop])
    return 0


def _exit_with_parent(parent_pid: int) -> None:
    """Stop this process (as SIGTERM would) once the process that launched it is gone.

    On Windows, ``start_new_session=True`` breaks ``os.getppid()`` identity with the
    launcher, so the old ``getppid() == parent_pid`` watch exited immediately (status 15).
    Poll whether ``parent_pid`` still exists instead.
    """
    def parent_alive() -> bool:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(parent_pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(parent_pid, 0)
        except OSError:
            return False
        return True

    def watch() -> None:
        while parent_alive():
            time.sleep(1.0)
        # raise_signal runs the SIGTERM handler on both systems; os.kill on Windows terminates
        # the process at once, skipping the stoppers.
        signal.raise_signal(signal.SIGTERM)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


class _EgoProcess:
    """One robot's Ego Runtime, Ego API and Ego-State view in a process of its own, as on a robot:
    nothing of it shares memory or OpenGL contexts with the world. Which runtime to run is the
    body's business: the embodiment registry names the `steward` subcommand. Its output is passed
    through."""

    def __init__(self, args, robot_url: str, out_dir: Path, *, embodiment: str, api_port: int,
                 task: str | None = None, mission_config: dict | None = None):
        self.embodiment = embodiment
        self.task = task
        self.mission_config = dict(mission_config or {})
        body = EMBODIMENTS[embodiment]
        cmd = [sys.executable, "-m", "steward.cli", "--root", str(args.root), body["ego"],
               "--robot-url", robot_url, "--api-port", str(api_port), "--out-dir", str(out_dir),
               "--parent-pid", str(os.getpid()), "--bind-wait", str(args.bind_wait)]
        if task is not None:
            cmd += ["--task", task]
        if self.mission_config:
            cmd += ["--mission-config", json.dumps(self.mission_config, ensure_ascii=False)]
        if body["ego"] == "ego":  # the shared runtime segments by itself and drives the named body
            cmd += ["--segmentation-config", args.segmentation_config, "--body", embodiment]
        # The child writes its piped stdout in the locale encoding (GBK on Chinese Windows) unless told
        # otherwise; the reader decodes UTF-8, so a locale decode error cannot kill it and backpressure the child.
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", env=os.environ | {"PYTHONIOENCODING": "utf-8"},
                                     start_new_session=True)
        first = self.proc.stdout.readline()
        if not first:
            raise SystemExit(f"the Ego Runtime process for {robot_url} ({body['ego']}) exited with status "
                             f"{self.proc.wait()} before starting")
        self.url = json.loads(first)["egoApi"]
        self._stopping = False
        self._reported = False
        threading.Thread(target=self._relay, name="ego-output", daemon=True).start()

    def _relay(self) -> None:
        for line in self.proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    def check(self) -> None:
        """Report once on stderr when the Ego Runtime process ended without being stopped."""
        if not self._stopping and not self._reported and self.proc.poll() is not None:
            self._reported = True
            print(f"steward up: the Ego Runtime process exited with status {self.proc.returncode}; "
                  "the robot no longer perceives or acts", file=sys.stderr, flush=True)

    def stop(self) -> None:
        self._stopping = True
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class EgoSupervisor:
    """`steward up`'s keeper of Ego Runtime processes: one per robot of the loaded scenario, of the
    kind its body needs.

    A scenario switch is two steps around the world's own swap. :meth:`depart` runs first, while
    the old world is still up, and stops every runtime that cannot survive the new scenario -- a
    runtime still polling when its world is replaced would see the clock go back and start a
    round against a scenario it is not built for. :meth:`reconcile` runs afterwards and starts
    the runtimes the new bodies need. Same robot, same body: the process is kept and reboots
    itself on the new episode. The console's 重置 replaces every process (:meth:`retire`, then
    :meth:`reconcile`), so nothing the robot believed, planned or recorded is left over.

    The first robot keeps the fixed Ego API port (the console chat and `steward chores --api`
    find it there); any further robot takes a free port. A robot whose body is not in the
    registry is left alone and said so: the world shows it, nothing drives it.
    """

    def __init__(self, args, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.procs: dict[str, _EgoProcess] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _wanted(scenario: dict) -> dict[str, tuple[str, str | None, dict]]:
        """Per robot: the body, the mission and its parameters the scenario asks for."""
        from steward.embodiments import scenario_embodiments

        wanted: dict[str, tuple[str, str | None, dict]] = {}
        for rid, body in scenario_embodiments(scenario).items():
            if not body["registered"]:
                continue
            runs_mission = EMBODIMENTS[body["id"]]["runs_mission"]
            wanted[rid] = (body["id"],
                           scenario.get("mission") if runs_mission else None,
                           dict(scenario.get("missionConfig") or {}) if runs_mission else {})
        return wanted

    @staticmethod
    def _departing(wanted: dict[str, tuple[str, str | None, dict]],
                   procs: dict[str, _EgoProcess]) -> list[str]:
        """Robots whose runtime is not what the incoming scenario asks for.

        A different body, a different mission, or different station parameters all mean the
        process must go: a runtime is built around its mission and keeps its own belief, so
        retasking it in place would leave it acting on the old scenario.
        """
        return [rid for rid, proc in procs.items()
                if wanted.get(rid) != (proc.embodiment, proc.task, proc.mission_config)]

    def depart(self, host, scenario: dict) -> None:
        """Stop the runtimes whose robot or body is not in the scenario about to load."""
        with self._lock:
            for rid in self._departing(self._wanted(scenario), self.procs):
                self.procs.pop(rid).stop()
                host.detach_ego_api(rid)

    def retire(self, host) -> None:
        """Stop every runtime; :meth:`reconcile` starts them anew."""
        with self._lock:
            procs, self.procs = self.procs, {}
            for rid, proc in procs.items():
                proc.stop()
                host.detach_ego_api(rid)

    def reconcile(self, host) -> dict[str, str]:
        with self._lock:
            wanted = self._wanted(host.scenario)
            for rid in self._departing(wanted, self.procs):
                self.procs.pop(rid).stop()
                host.detach_ego_api(rid)
            for rid, body in host.embodiments.items():
                if rid in self.procs:
                    continue
                if not body["registered"]:
                    print(f"steward up: {rid} is a {body['id']}, which is not a registered embodiment; "
                          f"no Ego Runtime drives it", file=sys.stderr, flush=True)
                    continue
                _, task, mission_config = wanted[rid]
                proc = _EgoProcess(self.args, host.robot_servers[rid].url, self.run_dir / f"ego-{rid.removeprefix('w:')}",
                                   embodiment=body["id"], api_port=self.args.api_port if not self.procs else 0,
                                   task=task, mission_config=mission_config)
                self.procs[rid] = proc
                host.attach_ego_api(rid, proc.url, skills=EMBODIMENTS[body["id"]]["skills"])
            if host.chat.config is not None:
                host.chat.config.ego_api = self.primary_url()
            return {rid: proc.url for rid, proc in self.procs.items()}

    def primary_url(self) -> str | None:
        return next((proc.url for proc in self.procs.values()), None)

    def check(self) -> None:
        with self._lock:
            procs = list(self.procs.values())
        for proc in procs:
            proc.check()

    def stop(self) -> None:
        with self._lock:
            procs = list(self.procs.values())
            self.procs.clear()
        for proc in procs:
            proc.stop()


def cmd_up(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    # The robot port serves from construction; the console's WCA opens only once the Ego APIs are
    # attached, so the console never asks for an Ego-State view no robot provides yet.
    host = _host(args, run_dir, start=False)
    egos = EgoSupervisor(args, run_dir)
    try:
        egos.reconcile(host)
    except BaseException:
        egos.stop()
        host.stop()
        raise
    # The console can load a scenario with another body: its departing robots' runtimes stop
    # before the world they poll is replaced, and the new bodies' runtimes follow.
    host.before_episode_load = lambda scenario: egos.depart(host, scenario)
    host.on_episode_loaded = lambda: egos.reconcile(host)
    host.before_episode_reset = lambda: egos.retire(host)
    host.start()
    primary = next(iter(host.scenario["robots"]), None)
    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url,
                      "robot": host.robot_servers[primary].url if primary else None, "egoApi": egos.primary_url(),
                      "robots": {rid: {"url": host.robot_servers[rid].url, "embodiment": body["id"],
                                       "egoApi": host.ego_apis.get(rid)} for rid, body in host.embodiments.items()},
                      "runDir": str(run_dir), "task": host.scenario["task"]}, ensure_ascii=False), flush=True)
    host.dispatch_scenario_task()

    def pump(timeout: float) -> None:
        host.pump(timeout)
        egos.check()

    _wait_forever([egos.stop, host.stop], pump)
    return 0


def cmd_demo(args) -> int:
    from steward.authoring.service import AuthoringService, build_default_kitchen
    from steward.mission import MissionFailed, cup_to_table

    if "phase0-kitchen" not in AuthoringService(args.root).list_scenarios():
        build_default_kitchen(AuthoringService(args.root))
    run_dir = Path(args.root) / "runs" / ("demo-" + time.strftime("%Y%m%d-%H%M%S"))
    host = _host(args, run_dir)
    runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego",
                              segmenter=_segmenter(args.segmentation_config, args.root))
    t0 = time.monotonic()
    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = cup_to_table(api.url)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="mission", daemon=True)
    worker.start()
    while worker.is_alive():
        host.pump(0.05)
    ok_mission = "result" in outcome
    result = outcome.get("result") or {"error": outcome.get("error")}
    time.sleep(1.0 / max(args.speed, 0.1))
    ws = host.sim.world_state()
    on_table = any(r["type"] == "supported-by" and r["a"]["entity"] == "w:cup-1" and r["b"]["entity"] == "w:table-1"
                   for r in ws["relations"])
    (run_dir / "overview-final.png").write_bytes(host.sim.render("overview", 800, 600))
    report = {
        "missionCompleted": ok_mission,
        "groundTruthCupOnTable": on_table,
        "wallSeconds": round(time.monotonic() - t0, 1),
        "simSeconds": round(host.sim.time, 1),
        "mission": result.get("log", result),
        "observer": host.observer.report(),
        "egoObjects": [(o["id"], o["category"], o["position"], o["uncertaintyM"], o["existence"])
                       for o in runtime.summary()["objects"]],
        "observerMatches": host.observer.latest.get("w:robot-1", {}).get("matches"),
        "events": [e for e in host.sim.events if e["type"].startswith("grasp/")],
        "runDir": str(run_dir),
    }
    api.stop()
    runtime.stop()
    host.stop()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok_mission and on_table else 1


def cmd_build_cobot(args) -> int:
    from steward.authoring.service import AuthoringService, build_cobot_workstation

    out = build_cobot_workstation(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_build_cobot_kitchen(args) -> int:
    from steward.authoring.service import AuthoringService, build_cobot_kitchen

    out = build_cobot_kitchen(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_build_cobot_errand(args) -> int:
    from steward.authoring.service import AuthoringService, build_cobot_errand

    out = build_cobot_errand(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_cobot(args) -> int:
    """Run the Cobot Magic twin with its console; the mission is driven by the robot's own perception."""
    from steward.assets.cobot import cobot_magic
    from steward.ego.cobot_runtime import MISSIONS
    from steward.ego.cobot_server import cobot_ego_api
    from steward.ego.cobot_skills import CobotSkills
    from steward.ego.estimator import EgoEstimator
    from steward.ego.perception import decode_rgbd
    from steward.phases import phase_modules

    default_scenario = {"nest-bowls": "cobot-kitchen", "carry-to-counter": "cobot-errand"}
    args.scenario = args.scenario or default_scenario.get(args.task, "cobot-table")
    run_dir = Path(args.root) / "runs" / ("cobot-" + time.strftime("%Y%m%d-%H%M%S"))
    host = _host(args, run_dir)  # builds the builtin scenario on first use
    sim = host.sim
    robot_id = host.scenario["robots"][0]
    skills = CobotSkills(command=lambda cmd: sim.command(robot_id, cmd), state=lambda: sim.sensors(robot_id))
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    rgbd = phase_modules(host.phase)["perception"] == "rgbd"
    # The console's Ego State panel is rendered from the mission's own estimator, so the operator
    # watches the belief the skills are acting on -- not a second, independent one. Its self part
    # reads the twin's encoders on every frame, so the arm on that panel moves with the arm.
    ego = cobot_ego_api(est, port=0, live=lambda: sim.proprioception(robot_id))
    host.attach_ego_api(robot_id, ego.url, skills=False)

    def sensed_frame() -> dict:
        return decode_rgbd(sim.rgbd(robot_id, "wide_camera")) if rgbd else sim.sensors(robot_id)

    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url,
                      "robot": host.robot_servers[robot_id].url, "egoApi": ego.url,
                      "task": host.scenario["task"], "phase": host.phase,
                      "perception": "ground-truth" if args.ground_truth else ("rgbd" if rgbd else "detections"),
                      "runDir": str(run_dir)}, ensure_ascii=False), flush=True)

    outcome: dict = {}
    homes = {ent.id: list(next(e for e in sim.world_state()["entities"] if e["id"] == ent.id)["pose"]["value"]["p"])
             for ent in sim.template.entities if _tableware(ent)}

    def reset_for_next_round() -> None:
        """Park both arms, put every piece of tableware back where the round started, and let the
        robot look again.

        Parking first is not cosmetic: a round ends with a hand over the object it just placed, and
        the head cameras then see a forearm instead of the tabletop -- the next round would
        start blind. The objects' moves are world edits by the round driver, so the belief is told
        to start over rather than being left to decide on its own that they teleported."""
        from steward.assets.cobot import park

        try:
            # Command the park and let physics settle rather than waiting for a skill to converge:
            # the point is only to get the forearms out of the head cameras' view, and the next
            # motion re-seeds its IK from the encoders anyway.
            for arm in ("L", "R"):
                sim.command(robot_id, {"joints": dict(zip(skills.ik.joint_names(arm), park(arm), strict=True))})
            for entity_id, at in homes.items():
                sim.edit_move(entity_id, {"at": {"x": at[0], "y": at[1], "z": at[2]}})
            sim.step_seconds(1.2)
        except Exception as exc:  # noqa: BLE001 - a stuck reset must not end the loop
            print(json.dumps({"reset": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
            return
        est.tracks.clear()
        sim.step(30)

    def run_truth_mission() -> None:
        """Escape hatch for debugging the skills: aim at the true object poses."""
        def pose_of(object_id: str) -> list[float]:
            state = sim.world_state()
            return list(next(e for e in state["entities"] if e["id"] == object_id)["pose"]["value"]["p"])

        bowl = pose_of("w:bowl-1")
        bowl[2] += 0.03
        plate = pose_of("w:plate-1")
        target = [plate[0], plate[1], plate[2] + 0.03]
        for name, plan in (("pick", {"arm": "L", "at": bowl}), ("place", {"arm": "L", "at": target})):
            job = skills.start(name, {"id": f"cli-{name}", "plan": plan})
            while job.state == "running":
                time.sleep(0.05)  # the main thread owns pump(); the job thread must not
            print(json.dumps({"skill": name, "state": job.state, "message": job.message,
                              "log": job.log}, ensure_ascii=False), flush=True)
            if job.state != "succeeded":
                outcome["error"] = job.message
                return
        outcome["done"] = True

    def run_sensed_mission() -> None:
        rounds = args.rounds
        mission = MISSIONS[args.task]
        # Station parameters are the scenario's: the same mission runs in a different kitchen by
        # loading a different scenario, not by editing code.
        kwargs = dict(host.scenario.get("missionConfig") or {})
        done, crashes = 0, 0
        while True:
            try:
                result = mission(est, skills, sensed_frame, rgbd=rgbd, **kwargs)
                outcome.update(result)
                print(json.dumps({"mission": args.task, "round": done + 1, **result},
                                 ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001 - reported to the operator, console stays up
                crashes += 1
                outcome["error"] = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"round": done + 1, **outcome}, ensure_ascii=False), flush=True)
                if args.rounds or crashes >= 3:
                    return  # a single-shot run reports the error; a loop stops only if it repeats
            else:
                crashes = 0
            done += 1
            if rounds and done >= rounds:
                return
            time.sleep(1.0)
            reset_for_next_round()

    if args.ground_truth and args.task not in ("bowl-on-plate", "nest-bowls"):
        raise SystemExit(f"steward cobot: --ground-truth is written against the table topologies "
                         f"(bowl/plate ids), not the {args.task} scenario")
    worker = threading.Thread(target=run_truth_mission if args.ground_truth else run_sensed_mission,
                              name="cobot-mission", daemon=True)
    worker.start()
    _wait_forever([ego.stop, host.stop], host.pump)
    return 0


def _ground_truth_locations(scene: dict, supports: list[str], objects: list[str]) -> dict[str, str | None]:
    """Support category under each carried object category, from World State relations."""
    category = {e["id"]: e["category"]["value"] for e in scene["entities"]}
    out: dict[str, str | None] = {cat: None for cat in objects}
    for rel in scene["relations"]:
        a, b = category.get(rel["a"]["entity"]), category.get(rel["b"]["entity"])
        if rel["type"] == "supported-by" and a in out and b in supports:
            out[a] = b
    return out


SUPPORT_NAMES = {"table": "餐桌", "cabinet": "柜子", "sideboard": "餐边柜", "counter": "操作台"}
OBJECT_NAMES = {"cup": "杯子", "bowl": "碗", "plate": "盘子"}


def cmd_shuttle(args) -> int:
    """Shuttle objects between table, cabinet and sideboard, checking each move against World State."""
    from steward.mission import MissionFailed, shuttle
    from steward.net import call

    supports, objects = ["table", "cabinet", "sideboard"], ["cup", "bowl"]
    host = runtime = api = None
    if args.api:
        api_url = args.api.rstrip("/")
        def world_scene():
            return call("GET", f"{args.wca.rstrip('/')}/wca/scene")
    else:
        from steward.authoring.service import AuthoringService, build_default_kitchen

        if args.scenario not in AuthoringService(args.root).list_scenarios():
            build_default_kitchen(AuthoringService(args.root))
        run_dir = Path(args.root) / "runs" / ("shuttle-" + time.strftime("%Y%m%d-%H%M%S"))
        host = _host(args, run_dir)
        runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego",
                                  segmenter=_segmenter(args.segmentation_config, args.root))
        api_url = api.url
        def world_scene():
            return host.sim.world_state()
        print(json.dumps({"console": f"{host.wca.url}/console"}), flush=True)
    t0 = time.monotonic()
    results: list[bool] = []

    def on_step(index: int, move: dict) -> None:
        time.sleep(0.5)
        truth = _ground_truth_locations(world_scene(), supports, objects)
        ok = truth.get(move["object"]) == move["to"]
        results.append(ok)
        if args.steps > 0:
            filled = int(30 * (index + 1) / args.steps)
            progress = f"[{'#' * filled}{'-' * (30 - filled)}] {index + 1}/{args.steps}"
        else:
            progress = f"[第 {index + 1} 步，第 {index // 3 + 1} 轮]"
        print(f"{progress}  "
              f"{OBJECT_NAMES.get(move['object'], move['object'])}：{SUPPORT_NAMES.get(move['from'], '其他位置')} → "
              f"{SUPPORT_NAMES[move['to']]}  World State {'确认' if ok else '不一致'} "
              f"{ {OBJECT_NAMES.get(k, k): SUPPORT_NAMES.get(v, v) for k, v in truth.items()} }  "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    def on_failure(index: int, message: str) -> None:
        print(f"[第 {index + 1} 步失败，重新观察后继续]  {message}  ({time.monotonic() - t0:.0f}s)", flush=True)

    def on_wait(owner: str) -> None:
        print(f"[暂停：机器人由 {owner} 接管，交还控制后继续]  ({time.monotonic() - t0:.0f}s)", flush=True)

    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = shuttle(api_url, args.steps, tuple(supports), tuple(objects), on_step=on_step,
                                        on_failure=on_failure, on_wait=on_wait)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="shuttle", daemon=True)
    worker.start()
    interrupted = False
    try:
        while worker.is_alive():
            if host is not None:
                host.pump(0.05)
            else:
                worker.join(0.2)
    except KeyboardInterrupt:
        interrupted = True
        if host is None:
            call("POST", f"{api_url}/skill/stop", {})
    finished = "result" in outcome or (interrupted and "error" not in outcome)
    ok = finished and (args.steps <= 0 or len(results) == args.steps) and all(results)
    print(json.dumps({"completedSteps": len(results), "allConfirmedByWorldState": ok,
                      "error": outcome.get("error")}, ensure_ascii=False), flush=True)
    if host is not None:
        api.stop()
        runtime.stop()
        host.stop()
    return 0 if ok else 1


def cmd_chores(args) -> int:
    """Repeat the chores rounds, checking every completed phase against World State."""
    from steward.chores import PHASE_NAMES, chores
    from steward.mission import MissionFailed
    from steward.net import call
    from steward.world.chores_check import check_sideboard, check_swap, check_table, drawer_contents

    host = runtime = api = None
    if args.api:
        api_url = args.api.rstrip("/")
        wca = args.wca.rstrip("/")

        def truth() -> tuple[dict, dict]:
            boxes = {g["id"]: g for g in call("GET", f"{wca}/console/state")["groundTruth"]}
            return call("GET", f"{wca}/wca/scene"), boxes
    else:
        from steward.authoring.service import AuthoringService, build_default_kitchen
        from steward.world.console import console_state

        if args.scenario not in AuthoringService(args.root).list_scenarios():
            build_default_kitchen(AuthoringService(args.root))
        run_dir = Path(args.root) / "runs" / ("chores-" + time.strftime("%Y%m%d-%H%M%S"))
        host = _host(args, run_dir)
        runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego",
                                  segmenter=_segmenter(args.segmentation_config, args.root))
        api_url = api.url

        def truth() -> tuple[dict, dict]:
            return host.sim.world_state(), {g["id"]: g for g in console_state(host)["groundTruth"]}
        print(json.dumps({"console": f"{host.wca.url}/console"}), flush=True)
    t0 = time.monotonic()
    results: list[bool] = []
    drawers_before = {"contents": drawer_contents(truth()[0])}

    def stamp() -> str:
        return f"({time.monotonic() - t0:.0f}s)"

    def on_step(record: dict) -> None:
        print(f"    · {record['text']}  {stamp()}", flush=True)

    def on_phase(index: int, phase: str) -> None:
        time.sleep(0.5)
        scene, boxes = truth()
        if phase == "table":
            ok, detail = check_table(scene, boxes)
        elif phase == "sideboard":
            ok, detail = check_sideboard(scene, boxes)
        else:
            ok, detail = check_swap(drawers_before["contents"], scene)
            drawers_before["contents"] = drawer_contents(scene)
        results.append(ok)
        print(f"[第 {index + 1} 轮] {PHASE_NAMES[phase]}完成  World State {'确认' if ok else '不一致'}：{detail}  {stamp()}",
              flush=True)

    def on_failure(index: int, phase: str, message: str) -> None:
        print(f"[第 {index + 1} 轮] {PHASE_NAMES[phase]}失败，放下手中物品、重新观察后继续：{message}  {stamp()}", flush=True)
        drawers_before["contents"] = drawer_contents(truth()[0])

    def on_wait(owner: str) -> None:
        print(f"[暂停：机器人由 {owner} 接管，交还控制后继续]  {stamp()}", flush=True)

    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = chores(api_url, args.rounds, on_step=on_step, on_phase=on_phase, on_failure=on_failure,
                                       on_wait=on_wait)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="chores", daemon=True)
    worker.start()
    interrupted = False
    try:
        while worker.is_alive():
            if host is not None:
                host.pump(0.05)
            else:
                worker.join(0.2)
    except KeyboardInterrupt:
        interrupted = True
        if host is None:
            call("POST", f"{api_url}/skill/stop", {})
    finished = "result" in outcome or (interrupted and "error" not in outcome)
    ok = finished and (args.rounds <= 0 or len(results) == 3 * args.rounds) and all(results)
    print(json.dumps({"completedPhases": len(results), "allConfirmedByWorldState": ok, "error": outcome.get("error")},
                     ensure_ascii=False), flush=True)
    if host is not None:
        api.stop()
        runtime.stop()
        host.stop()
    return 0 if ok else 1


def cmd_services(args) -> int:
    from steward import perception_services as ps

    services = ps.load(Path(args.config))
    healthy = True
    for svc in services.values():
        if args.no_check:
            print(json.dumps({"service": svc.name, "title": svc.title, "urls": list(svc.urls), "docs": svc.docs},
                             ensure_ascii=False))
            continue
        health = ps.check(svc)
        healthy &= health.ok
        print(json.dumps({"service": svc.name, "title": svc.title, "ok": health.ok, "url": health.url,
                          "detail": health.detail}, ensure_ascii=False), flush=True)
    return 0 if healthy else 1


def _speed(text: str) -> float:
    return math.inf if text == "max" else float(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steward")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Steward data root (assets, scenes, runs)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-kitchen").set_defaults(fn=cmd_build_kitchen)
    sub.add_parser("build-toy-assembly").set_defaults(fn=cmd_build_toy_assembly)
    sub.add_parser("build-screw-board").set_defaults(fn=cmd_build_screw_board)
    sub.add_parser("build-cobot").set_defaults(fn=cmd_build_cobot)
    sub.add_parser("build-cobot-kitchen").set_defaults(fn=cmd_build_cobot_kitchen)
    sub.add_parser("build-cobot-errand").set_defaults(fn=cmd_build_cobot_errand)
    defaults = {"demo": "phase0-kitchen", "shuttle": "phase0-shuttle"}
    for name, fn in (("world", cmd_world), ("up", cmd_up), ("demo", cmd_demo), ("shuttle", cmd_shuttle),
                     ("chores", cmd_chores), ("cobot", cmd_cobot)):
        p = sub.add_parser(name)
        # A cobot run picks its scenario from its task, so it has no fixed default here.
        p.add_argument("--scenario", default=None if name == "cobot" else defaults.get(name, "phase0-chores"))
        in_process = name in ("demo", "shuttle", "chores", "cobot")
        p.add_argument("--speed", type=_speed, nargs="?", const=math.inf, default=4.0 if in_process else 1.0,
                       help="simulation speed as a multiple of real time; without a value (or 'max'), as fast as possible")
        p.add_argument("--freeze-idle", action="store_true",
                       help="no world time passes while no robot acts, e.g. while the agents think")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--phase", choices=sorted(PHASES), default=None,
                       help="phase level of the replaceable modules for every episode (switchable at run time "
                            f"from the console); default: the scenario's declared level, else {LATEST_PHASE}")
        p.add_argument("--wca-port", type=int, default=COBOT_WCA_PORT if name == "cobot" else (0 if in_process else 7400))
        p.add_argument("--robot-port", type=int, default=COBOT_ROBOT_PORT if name == "cobot" else (0 if in_process else 7410))
        p.add_argument("--misnames-config", default=str(MISNAMES_CONFIG),
                       help="names the ideal detections report instead of true categories at the phase0-misnamed level; '' disables it")
        if not in_process:
            p.add_argument("--bind-wait", type=float, default=10.0,
                           help="seconds to wait for a fixed port still held by a service that just stopped")
            p.add_argument("--body", choices=list(EMBODIMENTS), default=None,
                           help="the body every episode's robot gets, whatever body its scenario was built with "
                                "(switchable from the console); default: each scenario's own")
            p.add_argument("--pause-on-perception-fault", action=argparse.BooleanOptionalAction, default=True,
                           help="pause the world when the Observer finds a gross perception fault (phantom object, "
                                "robot inside an object) until the operator acknowledges the alert")
            p.add_argument("--world-harness", default=str(STEWARD_DIR / "var" / "dsh" / "world"),
                           help="DSH home of the World Harness used by the console chat; '' disables it")
            p.add_argument("--robot-harness", default=str(STEWARD_DIR / "var" / "dsh" / "robot-1"),
                           help="DSH home of the robot's Ego Harness used by the console chat; '' disables it")
            p.add_argument("--turn-budget-s", type=float, default=TURN_BUDGET_S,
                           help="wall seconds one console chat turn of a harness may run before it is ended; "
                                "default: 24 days")
        if name == "world":
            p.add_argument("--ego-api", default="http://127.0.0.1:7420",
                           help="Ego API of the robot, for robot tasks from the console chat")
        if name == "up":
            p.add_argument("--api-port", type=int, default=7420)
            p.add_argument("--dispatch-task", action=argparse.BooleanOptionalAction, default=True,
                           help="start the robot's Ego Harness on the scenario's task, in a new conversation, "
                                "at start and whenever the console loads or resets an episode")
        if name != "world":
            p.add_argument("--segmentation-config", default=str(SEGMENTATION_CONFIG),
                           help="robot-side segmenter settings, used at phase levels without world masks; '' disables it")
        if name == "chores":
            p.add_argument("--rounds", type=int, default=1, help="number of rounds; 0 repeats until Ctrl-C")
            p.add_argument("--api", help="Ego API of a running `steward up`; omit to run in-process")
            p.add_argument("--wca", default="http://127.0.0.1:7400", help="WCA of the running world (with --api)")
        if name == "shuttle":
            p.add_argument("--steps", type=int, default=6, help="number of moves; 0 repeats until Ctrl-C")
        if name == "cobot":
            p.add_argument("--task", choices=COBOT_MISSIONS, default="bowl-on-plate",
                           help="what the twin does in the scene: bowl onto the plate, or nest the three "
                                "coloured bowls of the kitchen (the cobot-kitchen scenario)")
            p.add_argument("--ground-truth", action="store_true",
                           help="aim the skills at true object poses instead of the robot's estimates (debugging)")
            p.add_argument("--rounds", type=int, default=0,
                           help="rounds of bowl-on-plate to run; 0 repeats until Ctrl-C (default)")
            p.add_argument("--api", help="Ego API of a running `steward up`; omit to run in-process")
            p.add_argument("--wca", default="http://127.0.0.1:7400", help="WCA of the running world (with --api)")
        p.set_defaults(fn=fn)
    for name, fn in (("ego", cmd_ego), ("cobot-ego", cmd_cobot_ego)):
        p = sub.add_parser(name)
        p.add_argument("--robot-url", default="http://127.0.0.1:7410")
        p.add_argument("--api-port", type=int, default=7420)
        p.add_argument("--out-dir", help="where the runtime writes its observations (default: <root>/ego/observations)")
        p.add_argument("--parent-pid", type=int, help="exit when this process is gone (set by `steward up`)")
        p.add_argument("--bind-wait", type=float, default=10.0,
                       help="seconds to wait for a fixed port still held by a service that just stopped")
        if name == "ego":
            p.add_argument("--segmentation-config", default=str(SEGMENTATION_CONFIG),
                           help="robot-side segmenter settings, used at phase levels without world masks; '' disables it")
            p.add_argument("--body", choices=list(EMBODIMENTS), default="steward-bot",
                           help="the registered embodiment the runtime drives (its model, arm and motion settings)")
        if name == "cobot-ego":
            p.add_argument("--task", choices=COBOT_MISSIONS, default=None,
                           help="scripted mission the twin runs in its scene, round after round, each new "
                                "episode (the scenario's `mission`); omit to only perceive")
            p.add_argument("--mission-config", default=None,
                           help="JSON object of parameters for the mission (the scenario's "
                                "`missionConfig`: station poses and the like)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("services")
    p.add_argument("--config", default=str(STEWARD_DIR / "config" / "perception-services.toml"),
                   help="perception services file")
    p.add_argument("--no-check", action="store_true", help="only list the services, without contacting them")
    p.set_defaults(fn=cmd_services)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except OSError as exc:
        from steward.net import address_in_use

        if not address_in_use(exc):
            raise
        raise SystemExit(f"steward: a port is already in use ({exc}); stop the other run or pass "
                         f"different --wca-port / --robot-port / --api-port values") from None


if __name__ == "__main__":
    sys.exit(main())
