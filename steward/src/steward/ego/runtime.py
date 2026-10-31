"""Ego Runtime (ER): perception loop, Ego-State store, ESim receipts, skill execution and the
operator view of the Ego-State (rendered on its own thread, see :mod:`steward.viewloop`).

The runtime talks to the world only through the robot port (sensors, actuator commands,
telemetry). It never holds World State identifiers or ground truth.
"""

from __future__ import annotations

import collections
import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from steward.ego.classify import classify
from steward.ego.perception import (box_of, decode_rgbd, match_face, resample_face, segments, top_face, top_face_array,
                                    world_up_face)
from steward.ego.esim import Esim, EsimConfig
from steward.ego.estimator import SELF_ID, EgoEstimator
from steward.ego.roles import RoleConfig, assign_roles
from steward.ego.skills import MotionConfig, SkillExecutor
from steward.ego.view import EgoViewRenderer
from steward.math3d import wrap_angle, yaw_from_quat
from steward.net import HttpError, call
from steward.scene.schema import SceneState
from steward.viewloop import ViewLoop


@dataclass
class RuntimeConfig:
    perception_hz: float = 10.0
    telemetry_hz: float = 2.0
    appearance_cell_m: float = 0.005  # cell size of the remembered top faces
    appearance_max_m: float = 0.6  # larger objects' faces are remembered only when asked for
    appearance_range_m: float = 1.5  # faces farther from the camera are not looked at
    appearance_interval_s: float = 0.5  # world time between two automatic looks at one face
    appearance_poll_s: float = 1.0  # wall time between camera images fetched for faces with detections
    appearance_min_coverage: float = 0.25  # a view showing less of a face is not kept
    max_risk: float = 0.35
    esim: EsimConfig = field(default_factory=EsimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)


class RobotLink:
    """HTTP client of one robot port: the only path between the Ego System and the world."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.closed = False  # set on shutdown: clock waits stop retrying

    def sensors(self) -> dict:
        return call("GET", f"{self.url}/sensors")

    def rgbd(self) -> bytes:
        """One RGB-D frame (``npz``) when the world provides raw imaging instead of detections."""
        return call("GET", f"{self.url}/sensors/rgbd")

    def command(self, cmd: dict) -> None:
        call("POST", f"{self.url}/command", cmd)

    def telemetry(self, scene: dict) -> None:
        call("POST", f"{self.url}/telemetry", scene)

    def camera_png(self, width: int = 320, height: int = 240) -> bytes:
        return call("GET", f"{self.url}/camera/rgb?width={width}&height={height}")

    def wait_until(self, t: float | None = None, hold: str | None = None, rank: int = 0, pulls: bool = True) -> float:
        """Wait until the world clock reaches ``t`` (``None``: now) and, with ``hold``, the world lets
        this control loop or input run (:meth:`WorldSim.wait_until`); return the world time then."""
        body = {"hold": hold, "rank": rank, "pulls": pulls, **({} if t is None else {"until": t})}
        while True:
            try:
                return call("POST", f"{self.url}/clock/wait", body)["t"]
            except HttpError as exc:
                if exc.status != 504 or self.closed:
                    raise

    def arm(self, hold: str, t: float | None = None, rank: int = 0, pulls: bool = True) -> None:
        """Register ``hold`` at world time ``t`` (``None``: now) without waiting, so the world stops for
        it before its owner first waits."""
        try:
            call("POST", f"{self.url}/clock/wait", {"hold": hold, "rank": rank, "pulls": pulls, "timeoutS": 0.0,
                                                     **({} if t is None else {"until": t})})
        except HttpError as exc:
            if exc.status != 504:
                raise

    def sleep(self, seconds: float) -> float:
        """Let ``seconds`` of world time pass and return the world time then."""
        return self.wait_until(self.wait_until() + seconds)

    def release(self, hold: str) -> None:
        call("POST", f"{self.url}/clock/release", {"hold": hold})


ALERT_TEXT_MAX = 1500
QUERY_MIN_PROBABILITY = 0.2
MATCH_REFERENCE_CELLS_MAX = 96  # cells along a reference side for match_face
MATCH_MIN_COVERAGE = 0.5  # share of a face that must have been seen to match it
ACTIVITY_MAX = 300
PLAN_STEP_STATES = frozenset({"pending", "active", "done", "failed", "skipped"})

class EgoRuntime:
    def __init__(self, link: RobotLink, robot_model: dict, config: RuntimeConfig | None = None,
                 out_dir: Path | None = None):
        self.link = link
        self.config = config or RuntimeConfig()
        self.estimator = EgoEstimator(robot_model=robot_model)
        self.esim = Esim(robot_model, self.config.esim)
        self.out_dir = out_dir
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.receipts: dict[str, dict] = {}
        self.last_error: str | None = None
        self.control_owner: str | None = None
        self.perception_mode: str | None = None
        self.alerts: list[dict] = []
        self.plan: dict | None = None
        self._roles: dict[str, str] = {}  # the latest role assignment, kept unless clearly outweighed
        self.activity: collections.deque[dict] = collections.deque(maxlen=ACTIVITY_MAX)
        self._activity_ids = itertools.count(1)
        self._input_lock = threading.Lock()
        self.inputs: list[dict] = []  # Ego API requests as applied, with their world time (:meth:`apply`)
        self._alert_ids = itertools.count(1)
        self.executor = SkillExecutor(self.link.command, self._motion_state, self._mark_held, self.config.motion,
                                      shift=self._shift_track, on_error=lambda msg: self.alert(msg, "skill"),
                                      on_finish=self._skill_finished, clock=self.link)
        self._view_scene: tuple[dict, int, float] | None = None  # (ES dict, version, stamp) for the view
        self.face_error: str | None = None  # why the latest camera image for faces could not be fetched
        self._face_thread: threading.Thread | None = None
        renderer = EgoViewRenderer()
        self.view = ViewLoop("ego", lambda settings: self._render_view(renderer, settings), renderer.close)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start perception in a worker thread. The first frame is taken there too: RGB-D frames
        are rendered on the world's main thread, which may be the caller."""
        self._running = True
        self.link.arm("perception", pulls=False)
        self._thread = threading.Thread(target=self._loop, name="ego-perception", daemon=True)
        self._thread.start()
        self._face_thread = threading.Thread(target=self._face_loop, name="ego-faces", daemon=True)
        self._face_thread.start()
        self.view.start()

    def stop(self) -> None:
        self._running = False
        self.link.closed = True
        self.view.stop()
        for thread in (self._thread, self._face_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def _poll_once(self) -> None:
        frame = self.link.sensors()
        mode = "rgbd" if frame.get("imaging") == "rgbd" else "detections"
        if mode != self.perception_mode:
            with self._lock:
                if self.perception_mode is not None:
                    # The sensors changed kind (a phase switch): objects are perceived afresh with
                    # the new pipeline; what the gripper holds and the running skill are kept.
                    self.estimator.forget_objects()
                self.perception_mode = mode
        if mode == "rgbd":
            rgbd = decode_rgbd(self.link.rgbd())
            with self._lock:
                if rgbd["header"]["stamp"] < self.estimator.stamp - 1e-6:
                    self._restart()
                self.estimator.update_rgbd(rgbd)
                self._look_at_faces(rgbd)
                self._publish_view_scene()
            return
        with self._lock:
            if frame["stamp"] < self.estimator.stamp - 1e-6:
                self._restart()
            self.estimator.update(frame)
            self._publish_view_scene()

    def _face_loop(self) -> None:
        """With detections, which carry no colours, fetch a camera image every ``appearance_poll_s``
        wall seconds while an object whose face is remembered lies within range, and remember the
        faces it shows. It holds no world clock: perception and control never wait for it."""
        while self._running:
            time.sleep(self.config.appearance_poll_s)
            if self.perception_mode != "detections":
                continue
            with self._lock:
                cam_p = np.asarray(self.estimator.camera_pose()["p"], float)
                due = any(not t.held and t.footprint <= self.config.appearance_max_m
                          and float(np.linalg.norm(t.pos - cam_p)) <= self.config.appearance_range_m
                          for t in self.estimator.tracks.values())
            if not due:
                continue
            try:
                rgbd = decode_rgbd(self.link.rgbd())
            except (HttpError, OSError) as exc:
                self.face_error = f"camera image for faces unavailable: {exc}"
                continue
            with self._lock:
                self._look_at_faces(rgbd)
            self.face_error = None

    def _telemetry_scene(self) -> dict:
        """The Ego-State sent as telemetry, each object carrying its scene-level role as the state
        variable ``role`` (``src: inferred``) next to its perceived category."""
        roles = {o["id"]: (o["role"], o["roleConfidence"]) for o in self.summary()["objects"]}
        scene = self.ego_state().to_dict()
        for ent in scene["entities"]:
            if ent["id"] in roles:
                role, p = roles[ent["id"]]
                ent.setdefault("stateVars", {})["role"] = {"value": role, "unc": {"kind": "categorical", "p": {role: p}},
                                                           "src": "inferred"}
        return scene

    def _publish_view_scene(self) -> None:
        """Hand the view a copy of the Ego-State taken right after a perception update (the caller
        holds the lock), so rendering never waits for the next update to finish."""
        self._view_scene = (self.estimator.to_scene().to_dict(), self.estimator.version, self.estimator.stamp)

    def _render_view(self, renderer: EgoViewRenderer, settings) -> tuple[bytes, str, dict]:
        scene = self._view_scene
        if scene is None:
            with self._lock:
                self._publish_view_scene()
            scene = self._view_scene
        ego, version, stamp = scene
        return renderer.render(ego, settings), "image/jpeg", {"esVersion": version, "t": round(stamp, 3)}

    def _restart(self) -> None:
        """The world clock went backwards: a new episode started, which the robot experiences as a
        reboot. Forget the Ego-State, receipts and any running skill."""
        self.executor.cancel()
        self.estimator = EgoEstimator(robot_model=self.estimator.robot_model, config=self.estimator.config)
        self.receipts.clear()

    def _loop(self) -> None:
        """Perceive every ``1 / perception_hz`` seconds of world time, from the time the runtime started;
        the world waits for each update but does not step for it alone (``freeze_idle``)."""
        period = 1.0 / self.config.perception_hz
        telemetry_every = max(1, int(self.config.perception_hz / self.config.telemetry_hz))
        k = 0
        t: float | None = None
        while self._running:
            try:
                t = self.link.wait_until(None if t is None else t + period, "perception", pulls=False)
            except (HttpError, OSError) as exc:
                self.last_error = f"robot port unavailable: {exc}"
                time.sleep(period)
                continue
            try:
                self._poll_once()
                k += 1
                if k % telemetry_every == 0:
                    self.link.telemetry(self._telemetry_scene())
                self.last_error = None
            except (HttpError, OSError) as exc:
                self.last_error = f"robot port unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001 - perception must keep running; the operator is alerted
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.alert(f"感知循环出现异常：{type(exc).__name__}: {exc}", "runtime")
        try:
            self.link.release("perception")
        except (HttpError, OSError):
            pass  # the world is gone or restarting; its hold lease expires on its own

    def apply(self, request: dict, fn, at: float | None = None, then: float | None = None):
        """Apply one Ego API request (``method``, ``path``, ``query``, ``body``) at a world instant, after
        the control loops due then, and append it with that time to :attr:`inputs`. A replay
        (:func:`steward.ego.server.replay`) passes the recorded time ``at`` and the next one, ``then``,
        at which the world stops for the next request."""
        with self._input_lock:
            try:
                t = self.link.wait_until(at, "input", rank=1)
            except (HttpError, OSError):
                return fn()  # no world clock (the robot port is unreachable): applied now, not recorded
            try:
                return fn()
            finally:
                self.inputs.append({"t": t, **request})
                if then is None:
                    self.link.release("input")
                else:
                    self.link.arm("input", then, rank=1)

    # --------------------------------------------------------------- state
    def ego_state(self) -> SceneState:
        with self._lock:
            return self.estimator.to_scene()

    def _motion_state(self) -> dict:
        with self._lock:
            est = self.estimator
            entities = {t.id: {"p": t.pos.tolist()} for t in est.tracks.values()}
            return {"stamp": est.stamp, "odometry": dict(est.odometry), "joints": dict(est.joints),
                    "gripper": dict(est.gripper), "entities": entities, "toolOffset": self.esim.tool_offset}

    def _mark_held(self, track_id: str) -> str:
        with self._lock:
            return self.estimator.mark_held(track_id)

    def _shift_track(self, track_id: str, delta: list[float]) -> None:
        with self._lock:
            self.estimator.shift(track_id, delta)

    # ------------------------------------------------------------- Ego API
    def summary(self) -> dict:
        es = self.ego_state()
        me = es.entity(SELF_ID)
        p = me.pose.value
        held = next((r.a["entity"] for r in es.relations if r.type == "attached"), None)
        supports = {r.a["entity"]: r.b["entity"] for r in es.relations if r.type == "supported-by"}
        objects = []
        for ent in es.entities:
            if ent.id == SELF_ID:
                continue
            cov = ent.pose.unc["cov"]
            size = ent.rigid.links[0].geometry["collision"][0]["size"]
            objects.append({
                "id": ent.id,
                "category": ent.category.value,
                "categoryConfidence": round(ent.category.unc["p"].get(ent.category.value, 1.0), 3),
                "categories": ent.category.unc["p"],
                "position": [round(v, 3) for v in ent.pose.value["p"]],
                "yawDeg": round(math.degrees(yaw_from_quat(ent.pose.value["q"])), 1),
                "uncertaintyM": round(math.sqrt((cov[0] + cov[7]) / 2), 4),
                "sizeM": [round(v, 3) for v in size],
                "on": supports.get(ent.id),
                "existence": ent.existence,
                "lastSeenAgoS": round(es.stamp - (ent.pose.t or es.stamp), 2),
                "affordances": [a.value for a in ent.rigid.links[0].affordances],
                "color": ent.state_vars["color"].value if "color" in ent.state_vars else None,
                "extentM": self._extent(ent.id),
                **({"openingM": round(self._opening(ent.id), 3)} if ent.category.value == "drawer" else {}),
                **self._front(ent),
                **self._appearance(ent.id),
            })
        with self._lock:
            roles = assign_roles(objects, self.config.roles, self._roles)
            self._roles = {oid: role for oid, (role, _) in roles.items()}
        for obj in objects:
            obj["role"], confidence = roles[obj["id"]]
            obj["roleConfidence"] = round(confidence, 3)
        job = self.executor.current.to_dict() if self.executor.current else None
        dropped = self.dropped()[-20:]
        return {
            "version": es.version,
            "t": round(es.stamp, 3),
            "self": {"x": round(p["p"][0], 3), "y": round(p["p"][1], 3),
                     "yawDeg": round(math.degrees(yaw_from_quat(p["q"])), 1),
                     "holding": held, "joints": {j.id: round(j.q.value[0], 3) for j in me.rigid.joints if j.q.value}},
            "objects": objects,
            "dropped": dropped,
            "currentSkill": job,
            "perception": self.perception_mode,
            "runtimeError": self.last_error,
            "faceCaptureError": self.face_error,
        }

    def dropped(self) -> list[dict]:
        """Every recent removal from the Ego-State, oldest first, with its reason."""
        with self._lock:
            return list(self.estimator.dropped)

    def _extent(self, track_id: str) -> list[float] | None:
        """Estimated physical size of a track, without the uncertainty margin added to ``sizeM``."""
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return [round(float(v), 3) for v in track.size] if track is not None else None

    def _front(self, ent) -> dict:
        """Heading of the face looking into the room, for furniture and drawers."""
        affordances = [a.value for a in ent.rigid.links[0].affordances]
        if "support-surface" not in affordances and "pullable" not in affordances:
            return {}
        with self._lock:
            track = self.estimator.tracks.get(ent.id)
            if track is None:
                return {}
            box = self.estimator.front(track)
        return {"frontDeg": round(math.degrees(box.yaw), 1), "frontHalfM": round(box.hx, 3)}

    def _opening(self, track_id: str) -> float:
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return self.estimator.drawer_opening(track) if track is not None else 0.0

    def query(self, category: str | None = None, near: str | None = None, radius: float | None = None) -> list[dict]:
        """Objects matching a category by role or, failing that, by probability (best match first),
        optionally within ``radius`` of another object."""
        objs = self.summary()["objects"]
        if category:
            score = lambda o: 2.0 if o["role"] == category else o["categories"].get(category, 0.0)  # noqa: E731
            objs = sorted((o for o in objs if score(o) >= QUERY_MIN_PROBABILITY), key=score, reverse=True)
        if near:
            anchor = next((o for o in self.summary()["objects"] if o["id"] == near), None)
            if anchor is None:
                raise ValueError(f"unknown entity {near!r}")
            r = radius if radius is not None else 1.0
            objs = [o for o in objs if o["id"] != near and math.dist(o["position"][:2], anchor["position"][:2]) <= r]
        return objs

    def inspect(self, entity: str, px: int = 24) -> dict:
        """The top face of an Ego-State object as the head camera sees it now, resampled to a
        ``px`` × ``px`` grid in the object's own frame: column = local +x to the right, row 0 = local
        +y (the far edge seen from above). Each cell is ``[r, g, b]`` or ``None`` where the face is
        out of view or hidden by something nearer. ``worldRows`` is the same grid turned
        ``worldTurnDeg`` counter-clockwise (:func:`world_up_face`) so that row 0 faces the map's +y.
        Reads one fresh RGB-D frame; no World State.

        :param entity: Ego-State object id.
        :param px: grid size, 4 to 64.
        :returns: ``{entity, px, yawDeg, sizeM, coverage, rows, worldRows, worldTurnDeg, stamp}``.
        :raises KeyError: unknown object; :raises ValueError: ``px`` out of range."""
        if not 4 <= px <= 64:
            raise ValueError("px must be between 4 and 64")
        with self._lock:
            track = self.estimator.tracks.get(entity)
            if track is None:
                raise KeyError(f"unknown object {entity!r}")
            centre, size, yaw, top = track.pos.copy(), track.size.copy(), float(track.yaw), track.top()
        frame = decode_rgbd(self.link.rgbd())
        with self._lock:
            cam = self.estimator.camera_pose(frame["header"]["odometry"])
            self._look_at_faces(frame, [entity])
        rows = top_face(frame, cam, centre, size, yaw, top, px)
        seen = sum(cell is not None for row in rows for cell in row)
        world_rows, turn = world_up_face(rows, yaw)
        return {"entity": entity, "px": px, "yawDeg": round(math.degrees(yaw), 2),
                "sizeM": [round(float(v), 3) for v in size], "coverage": round(seen / (px * px), 3),
                "rows": rows, "worldRows": world_rows, "worldTurnDeg": turn, "stamp": frame["header"]["stamp"]}

    def _look_at_faces(self, frame: dict, ids: list[str] | None = None) -> None:
        """Remember the top faces one decoded RGB-D frame shows (the caller holds the lock): of
        ``ids`` of any size, or of every object no larger than ``appearance_max_m`` whose face was
        not looked at for ``appearance_interval_s``. Only objects within ``appearance_range_m`` are
        looked at (a distant view would overwrite a sharp one), and a view showing less than
        ``appearance_min_coverage`` of a face is not kept."""
        cfg, est = self.config, self.estimator
        cam = est.camera_pose(frame["header"]["odometry"])
        cam_p = np.asarray(cam["p"], float)
        stamp = float(frame["header"]["stamp"])
        for track in list(est.tracks.values()) if ids is None else [est.tracks[i] for i in ids if i in est.tracks]:
            if track.held:
                continue
            known = track.appearance
            if float(np.linalg.norm(track.pos - cam_p)) > cfg.appearance_range_m:
                continue
            if ids is None and (track.footprint > cfg.appearance_max_m
                                or (known is not None and not known.stale and stamp - known.stamp < cfg.appearance_interval_s)):
                continue
            cols, rows = est.face_grid(track, cfg.appearance_cell_m)
            rgb = top_face_array(frame, cam, track.pos, track.size, float(track.yaw), track.top(), cols, rows)
            if np.isfinite(rgb[..., 0]).mean() >= cfg.appearance_min_coverage:
                est.remember_face(track.id, rgb, stamp)

    def _appearance(self, entity: str) -> dict:
        with self._lock:
            track = self.estimator.tracks.get(entity)
            known = track.appearance if track is not None else None
            if known is None:
                return {"appearance": None}
            return {"appearance": {"seenAgoS": round(self.estimator.stamp - known.stamp, 1),
                                   "coverage": round(known.coverage, 2), "spread": round(known.spread, 1),
                                   "stale": known.stale}}

    def match_face(self, entities: list[str], reference: str, px: int = 12) -> dict:
        """Where the top face of each of ``entities`` appears on the top face of ``reference``
        printed at the same scale (puzzle pieces on the picture of their box, tiles on a pattern),
        from the faces the robot remembers (:meth:`_look_at_faces`), refreshed first with one fresh
        RGB-D frame for those in view. Each face is resampled to ``px`` cells across its shorter
        side and the reference to cells of the same size, then compared by :func:`match_face`.

        :param entities: Ego-State ids of the objects whose faces are sought.
        :param reference: Ego-State id of the object showing the whole picture.
        :param px: cells across each object's shorter side, 4 to 16.
        :returns: ``{reference, referenceCoverage, results, stamp}``; each result holds ``entity``,
            ``cellM``, ``spread``, ``faceCoverage``, ``seenAgoS`` and ``matches`` (``offsetM``: the
            face centre's place in the reference frame from its centre; ``turnDeg``: quarter turns
            counter-clockwise that align the face with the reference; ``placeYawDeg``: the object
            heading that does so with the reference's heading; ``score``: correlation, 1 =
            identical), or ``entity`` and ``error`` when its face is not known well enough.
        :raises KeyError: unknown object; :raises ValueError: ``px`` out of range or the reference
            face not known well enough."""
        if not isinstance(entities, list) or not entities or not all(isinstance(e, str) for e in entities):
            raise ValueError("entities must be a non-empty list of object ids")
        if not 4 <= px <= 16:
            raise ValueError("px must be between 4 and 16")
        frame = decode_rgbd(self.link.rgbd())
        with self._lock:
            est = self.estimator
            missing = [eid for eid in [*entities, reference] if eid not in est.tracks]
            if missing:
                raise KeyError(f"unknown object {missing[0]!r}")
            self._look_at_faces(frame, [*entities, reference])
            ref_track = est.tracks[reference]
            ref = ref_track.appearance
            if ref is None or ref.stale or ref.coverage < MATCH_MIN_COVERAGE:
                raise ValueError(f"the face of {reference} is not known well enough "
                                 f"({0 if ref is None else round(ref.coverage, 2)} seen); look at it from within about a metre")
            ref_rgb, ref_size, ref_yaw = ref.rgb.copy(), ref_track.size.copy(), float(ref_track.yaw)
            faces = {}
            for eid in entities:
                track = est.tracks[eid]
                known = track.appearance
                if known is None or known.stale:
                    faces[eid] = "its face has not been seen since it was last put down; look at it from within about a metre"
                elif known.coverage < MATCH_MIN_COVERAGE:
                    faces[eid] = f"only {known.coverage:.0%} of its face has been seen; look at it from closer or another side"
                else:
                    faces[eid] = (known.rgb.copy(), track.size.copy(), round(est.stamp - known.stamp, 1), known.spread)
        results = []
        for eid in entities:
            face = faces[eid]
            if isinstance(face, str):
                results.append({"entity": eid, "error": face})
                continue
            rgb, size, ago, spread = face
            cell = float(min(size[0], size[1])) / px
            ox, oy = max(1, round(size[0] / cell)), max(1, round(size[1] / cell))
            rx, ry = round(ref_size[0] / cell), round(ref_size[1] / cell)
            if rx <= max(ox, oy) or ry <= max(ox, oy):
                results.append({"entity": eid, "error": f"{reference} is not larger than {eid}"})
                continue
            if max(rx, ry) > MATCH_REFERENCE_CELLS_MAX:
                results.append({"entity": eid, "error": f"{reference} spans more than {MATCH_REFERENCE_CELLS_MAX} "
                                                        f"cells of {cell:.3f} m; use a smaller px"})
                continue
            found = match_face(resample_face(rgb, ox, oy), resample_face(ref_rgb, rx, ry))
            matches = [{"offsetM": [round(float((m["col"] - rx / 2) * ref_size[0] / rx), 3),
                                    round(float((ry / 2 - m["row"]) * ref_size[1] / ry), 3)],
                        "turnDeg": 90 * m["turns"],
                        "placeYawDeg": round(math.degrees(wrap_angle(ref_yaw + m["turns"] * math.pi / 2)), 1),
                        "score": m["score"]} for m in found["matches"]]
            results.append({"entity": eid, "cellM": round(cell, 4), "spread": round(spread, 1),
                            "faceCoverage": round(float(np.isfinite(rgb[..., 0]).mean()), 2), "seenAgoS": ago,
                            "matches": matches})
        return {"reference": reference, "referenceCoverage": round(float(np.isfinite(ref_rgb[..., 0]).mean()), 2),
                "results": results, "stamp": frame["header"]["stamp"]}

    def observe(self) -> dict:
        frame = self.link.sensors()
        if frame.get("imaging") == "rgbd":
            rgbd = decode_rgbd(self.link.rgbd())
            with self._lock:
                cam = self.estimator.camera_pose()
            visible = []
            for seg in segments(rgbd, cam):
                centre, size, _ = box_of(seg.points)
                probs = classify(size, centre[2] - size[2] / 2, seg.color, seg.invalid_frac)
                best = max(probs, key=probs.get)
                visible.append({"category": best, "confidence": round(probs[best], 2), "rangeM": round(seg.range, 2)})
        else:
            visible = [{"category": d["category"], "rangeM": round(d["range"], 2)} for d in frame["detections"]]
        out = {"t": frame["stamp"], "visible": visible}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"observe-{frame['stamp']:.2f}.png"
            path.write_bytes(self.link.camera_png())
            out["imagePath"] = str(path)
        return out

    def predict(self, skill: str, params: dict) -> dict:
        es = self.ego_state()
        receipt = self.esim.predict(es, skill, params)
        receipt["withinRiskLimit"] = receipt["feasible"] and receipt["risk"] <= self.config.max_risk
        receipt["maxRisk"] = self.config.max_risk
        with self._lock:
            self.receipts[receipt["id"]] = receipt
        return receipt

    def annotate(self, request: dict) -> dict:
        """Name an Ego-State object as the robot's agent identified it (``label``, ``affordances``).

        :param request: ``{entity, label?, affordances?}``.
        :returns: the object's summary entry.
        :raises ValueError: malformed fields; :raises KeyError: no such object."""
        entity, label, affordances = request.get("entity"), request.get("label"), request.get("affordances")
        if not isinstance(entity, str):
            raise ValueError("annotate needs an entity id")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError("label must be a non-empty string")
        if affordances is not None and (not isinstance(affordances, list) or not all(isinstance(a, str) for a in affordances)):
            raise ValueError("affordances must be a list of strings")
        with self._lock:
            if entity not in self.estimator.tracks:
                raise KeyError(f"unknown object {entity!r}")
            self.estimator.annotate(entity, label, affordances)
            self._publish_view_scene()
        return next(o for o in self.summary()["objects"] if o["id"] == entity)

    def receipt(self, receipt_id: str) -> dict:
        if receipt_id not in self.receipts:
            raise KeyError(f"unknown receipt {receipt_id!r}")
        return self.receipts[receipt_id]

    # ------------------------------------------------------------------ alerts
    def alert(self, message: str, source: str, level: str = "error", goal: str = "", step: str = "",
              diagnosis: str = "") -> dict:
        """Record a situation the robot cannot resolve by itself (an unexpected exception, or a
        problem its agent or mission reports) with the goal and step it was pursuing and the
        reporter's own diagnosis. A goal or step left empty is taken from the published plan. A
        repeat of an unacknowledged message only bumps its count."""
        message = message[:ALERT_TEXT_MAX]
        with self._lock:
            plan = self.plan or {}
            for existing in reversed(self.alerts):
                if not existing["acknowledged"] and existing["message"] == message:
                    existing["count"] += 1
                    existing["t"] = round(self.estimator.stamp, 2)
                    return existing
            record = {"id": next(self._alert_ids), "t": round(self.estimator.stamp, 2), "level": level,
                      "source": source, "message": message, "count": 1, "acknowledged": False,
                      "goal": (goal or plan.get("goal", ""))[:ALERT_TEXT_MAX],
                      "step": (step or plan.get("current", ""))[:ALERT_TEXT_MAX],
                      "diagnosis": diagnosis[:ALERT_TEXT_MAX]}
            self.alerts.append(record)
            del self.alerts[:-50]
        self._log("alert", message)
        return record

    def alerts_view(self) -> dict:
        with self._lock:
            return {"alerts": [dict(a) for a in self.alerts],
                    "active": [dict(a) for a in self.alerts if not a["acknowledged"]]}

    def acknowledge(self, alert_id: int | None = None) -> dict:
        with self._lock:
            for record in self.alerts:
                if alert_id is None or record["id"] == alert_id:
                    record["acknowledged"] = True
        return self.alerts_view()

    # ---------------------------------------------------------------- activity
    def _log(self, kind: str, text: str) -> None:
        """Append to the activity record the console shows below the Ego-State view: ``step``,
        ``difficulty``, ``state``, ``skill`` and ``alert`` entries, newest last."""
        with self._lock:
            self.activity.append({"id": next(self._activity_ids), "t": round(self.estimator.stamp, 1),
                                  "kind": kind, "text": text[:ALERT_TEXT_MAX]})

    def _skill_finished(self, job) -> None:
        result = {"succeeded": "完成", "failed": "失败", "stopped": "被停止"}.get(job.state, job.state)
        self._log("skill", f"{job.skill} {result}" + (f"：{job.message}" if job.state != "succeeded" else ""))

    def activity_since(self, since: int = 0) -> list[dict]:
        """Activity entries with an id above ``since``, oldest first.

        :param since: the id of the newest entry the caller already has.
        :returns: the newer entries.
        """
        with self._lock:
            return [dict(e) for e in self.activity if e["id"] > since]

    # -------------------------------------------------------------------- plan
    def set_plan(self, plan: dict) -> dict:
        """Publish what the robot is trying to achieve: a goal, its steps with their status, the
        step in progress and the latest difficulty. A client (a mission script, the robot agent)
        owns its content; the runtime only stores the latest one and stamps it."""
        goal = plan.get("goal")
        steps = plan.get("steps", [])
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("a plan needs a goal")
        if not isinstance(steps, list) or not all(isinstance(st, dict) and isinstance(st.get("name"), str)
                                                   and st.get("status") in PLAN_STEP_STATES for st in steps):
            raise ValueError(f"plan steps need a name and a status in {sorted(PLAN_STEP_STATES)}")
        record = {"goal": goal[:300], "source": str(plan.get("source", "agent"))[:60],
                  "steps": [{"name": st["name"][:200], "status": st["status"]} for st in steps[:30]],
                  "round": plan.get("round"), "current": str(plan.get("current") or "")[:400],
                  "difficulty": str(plan.get("difficulty") or "")[:600], "state": str(plan.get("state") or "running")[:40],
                  "t": round(self.estimator.stamp, 2)}
        with self._lock:
            previous, self.plan = self.plan or {}, record
        if record["current"] and record["current"] != previous.get("current"):
            self._log("step", record["current"])
        if record["difficulty"] and record["difficulty"] != previous.get("difficulty"):
            self._log("difficulty", record["difficulty"])
        if record["state"] != previous.get("state", "running"):
            self._log("state", record["state"])
        return record

    def plan_view(self, since: int = 0) -> dict:
        """The latest published plan (``None`` before any), the skill running now and the activity
        entries newer than ``since``.

        :param since: the id of the newest activity entry the caller already has.
        :returns: ``{"plan", "skill", "activity", "activityLast", "t"}`` (``activityLast`` is the newest
            id, lower than ``since`` after the runtime restarted).
        """
        running = next((j for j in reversed(list(self.executor.jobs.values())) if j.state == "running"), None)
        with self._lock:
            plan = dict(self.plan) if self.plan else None
        with self._lock:
            newest = self.activity[-1]["id"] if self.activity else 0
        return {"plan": plan, "skill": running.to_dict() if running else None, "activity": self.activity_since(since),
                "activityLast": newest, "t": round(self.estimator.stamp, 2)}

    # ----------------------------------------------------------------- control
    def control(self) -> dict:
        return {"owner": self.control_owner}

    def acquire_control(self, owner: str) -> dict:
        """Give ``owner`` exclusive use of skills until it releases control; the running skill stops.
        Clients that are not the owner cannot start skills meanwhile, but anyone may still stop."""
        if not owner:
            raise ValueError("control needs an owner name")
        with self._lock:
            if self.control_owner not in (None, owner):
                raise PermissionError(f"the robot is already controlled by {self.control_owner}")
            self.control_owner = owner
        self.executor.stop()
        return self.control()

    def release_control(self, owner: str) -> dict:
        with self._lock:
            if self.control_owner == owner:
                self.control_owner = None
        return self.control()

    def _check_control(self, owner: str | None) -> None:
        if self.control_owner is not None and owner != self.control_owner:
            raise PermissionError(f"the robot is controlled by {self.control_owner}; wait until it releases control")

    def run(self, receipt_id: str, owner: str | None = None) -> dict:
        """Execute a predicted plan. The runtime re-checks the receipt here, where the motion happens."""
        self._check_control(owner)
        receipt = self.receipt(receipt_id)
        now = self.estimator.stamp
        if receipt.get("used"):
            raise PermissionError(f"receipt {receipt_id} was already executed; predict again")
        if not receipt["feasible"]:
            raise PermissionError(f"receipt {receipt_id} is not feasible: {'; '.join(receipt['issues'])}")
        if now > receipt["expiresAt"]:
            raise PermissionError(f"receipt {receipt_id} expired at t={receipt['expiresAt']:.1f}s; predict again")
        if receipt["risk"] > self.config.max_risk:
            raise PermissionError(f"receipt risk {receipt['risk']:.2f} exceeds the limit {self.config.max_risk:.2f}")
        receipt["used"] = True
        job = self.executor.start(receipt["skill"], receipt).to_dict()
        params = receipt.get("params") or {}
        target = params.get("target") or params.get("on") or (f"({params['x']:.2f}, {params['y']:.2f})" if "x" in params else "")
        self._log("skill", f"{receipt['skill']} 开始" + (f"：{target}" if target else ""))
        return job

    def explore(self, owner: str | None = None) -> dict:
        """Look around in place. Rotation without translation needs no receipt."""
        self._check_control(owner)
        return self.executor.start("look_around", {"id": "look-around", "plan": {}}).to_dict()

    def job(self, job_id: str) -> dict:
        if job_id not in self.executor.jobs:
            raise KeyError(f"unknown job {job_id!r}")
        return self.executor.jobs[job_id].to_dict()

    def wait(self, job_id: str, timeout: float = 120.0) -> dict:
        return self.executor.wait(job_id, timeout).to_dict()

    def stop_motion(self) -> dict:
        return self.executor.stop()
