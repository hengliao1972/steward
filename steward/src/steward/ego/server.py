"""Ego API: the robot-local HTTP interface the Ego Harness tools call.

Every response is derived from the Ego-State and the robot's own receipts and jobs. A client may
take exclusive control (``/ego/control``); skill and explore requests then carry ``owner``.
``/ego/view`` sets the camera, rate and size of the rendered Ego-State view and
``/ego/view/frame`` returns its newest frame (``204`` when none is newer than ``after``).
Requests that can change the robot's state are applied at a world instant and recorded with its
time (``/ego/inputs``); :func:`replay` applies such a recording again, exactly, at any speed.
"""

from __future__ import annotations

from steward.ego.runtime import EgoRuntime
from steward.net import NO_CONTENT, HttpError, JsonServer
from steward.viewloop import ViewSettings, frame_response


# Requests that neither change the robot's state nor draw on the world's randomness; all others are
# applied at a world instant and recorded (:meth:`EgoRuntime.apply`).
UNRECORDED = frozenset({
    ("GET", "/ego/state"), ("GET", "/ego/control"), ("GET", "/ego/alerts"), ("GET", "/ego/dropped"), ("GET", "/ego/view"),
    ("POST", "/ego/view"),
    ("GET", "/ego/view/frame"), ("GET", "/ego/plan"), ("GET", "/ego/receipt"), ("GET", "/skill/status"),
    ("POST", "/skill/wait"), ("POST", "/clock/sleep"), ("GET", "/ego/inputs"),
})


def ego_api(runtime: EgoRuntime, port: int = 0, bind_wait_s: float = 0.0) -> JsonServer:
    def guarded(route, fn):
        def wrapped(q, b):
            try:
                return fn(q, b or {})
            except HttpError:
                raise
            except KeyError as exc:
                raise HttpError(404, str(exc).strip("'")) from None
            except (PermissionError, RuntimeError, ValueError) as exc:
                raise HttpError(409, str(exc)) from None
            except Exception as exc:  # noqa: BLE001 - reported to the operator, then to the caller as 500
                runtime.alert(f"Ego API {route[0]} {route[1]} 出现异常：{type(exc).__name__}: {exc}", "api")
                raise HttpError(500, f"{type(exc).__name__}: {exc}") from None
        return wrapped

    def recorded(route, fn):
        if route in UNRECORDED:
            return fn
        method, path = route
        return lambda q, b: runtime.apply({"method": method, "path": path, "query": q, "body": b}, lambda: fn(q, b))

    routes = _routes(runtime)
    return JsonServer({k: guarded(k, recorded(k, v)) for k, v in routes.items()}, port=port, name="ego-api",
                      bind_wait_s=bind_wait_s)


def replay(runtime: EgoRuntime, inputs: list[dict]) -> None:
    """Apply recorded Ego API requests (:attr:`EgoRuntime.inputs`) again, each at its recorded world
    time. With the same scenario, seed and phase, the runtime started at the same world time and the
    first request armed before the world runs on (``runtime.link.arm("input", inputs[0]["t"], rank=1)``),
    the run repeats exactly, at any speed."""
    routes = _routes(runtime)
    for i, request in enumerate(inputs):
        fn = routes[(request["method"], request["path"])]
        then = inputs[i + 1]["t"] if i + 1 < len(inputs) else None
        try:
            runtime.apply({k: request[k] for k in ("method", "path", "query", "body")},
                          lambda: fn(request["query"], request["body"] or {}), at=request["t"], then=then)
        except Exception:  # noqa: BLE001 - the request failed when it was recorded too; its effects are what count
            pass


def _routes(runtime: EgoRuntime) -> dict:
    def report(b):
        message = str(b.get("message", "")).strip()
        if not message:
            raise ValueError("an alert needs a message")
        return runtime.alert(message, str(b.get("source", "agent")), str(b.get("level", "blocked")),
                             goal=str(b.get("goal") or ""), step=str(b.get("step") or ""),
                             diagnosis=str(b.get("diagnosis") or ""))

    def view_frame(q):
        frame = runtime.view.latest(int(q["after"]) if q.get("after") else None)
        if frame is not None:
            return frame_response(frame, runtime.view.status())
        status = runtime.view.status()
        if status["settings"] is None:
            raise HttpError(409, "the ego view is not configured; POST /ego/view first")
        if status["error"] and status["frame"] is None:
            raise HttpError(503, f"the ego view cannot render: {status['error']}")
        return NO_CONTENT

    def control(b):
        if b.get("action") == "acquire":
            return runtime.acquire_control(b.get("owner", ""))
        if b.get("action") == "release":
            return runtime.release_control(b.get("owner", ""))
        raise ValueError("action must be acquire or release")

    return {
        ("GET", "/ego/summary"): lambda q, b: runtime.summary(),
        ("GET", "/ego/state"): lambda q, b: runtime.ego_state().to_dict(),
        ("GET", "/ego/query"): lambda q, b: {"objects": runtime.query(
            q.get("category"), q.get("near"), float(q["radius"]) if "radius" in q else None)},
        ("POST", "/ego/observe"): lambda q, b: runtime.observe(),
        ("POST", "/ego/explore"): lambda q, b: runtime.explore(b.get("owner")),
        ("GET", "/ego/control"): lambda q, b: runtime.control(),
        ("GET", "/ego/alerts"): lambda q, b: runtime.alerts_view(),
        ("GET", "/ego/dropped"): lambda q, b: {"dropped": runtime.dropped()},
        ("POST", "/ego/alert"): lambda q, b: report(b),
        ("POST", "/ego/alerts/ack"): lambda q, b: runtime.acknowledge(b.get("id")),
        ("GET", "/ego/view"): lambda q, b: runtime.view.status(),
        ("POST", "/ego/view"): lambda q, b: runtime.view.configure(ViewSettings.parse(b)),
        ("GET", "/ego/view/frame"): lambda q, b: view_frame(q),
        ("GET", "/ego/plan"): lambda q, b: runtime.plan_view(int(q.get("since", 0))),
        ("POST", "/ego/plan"): lambda q, b: runtime.set_plan(b),
        ("POST", "/ego/control"): lambda q, b: control(b),
        ("POST", "/ego/predict"): lambda q, b: runtime.predict(b["skill"], b.get("params", {})),
        ("POST", "/ego/annotate"): lambda q, b: runtime.annotate(b),
        ("POST", "/ego/inspect"): lambda q, b: runtime.inspect(b["entity"], int(b.get("px", 24))),
        ("POST", "/ego/match-face"): lambda q, b: runtime.match_face(b.get("entities"), b["reference"], int(b.get("px", 12))),
        ("GET", "/ego/receipt"): lambda q, b: runtime.receipt(q["id"]),
        ("POST", "/skill/run"): lambda q, b: runtime.run(b["receipt"], b.get("owner")),
        ("GET", "/skill/status"): lambda q, b: runtime.job(q["job"]),
        ("POST", "/skill/wait"): lambda q, b: runtime.wait(b["job"], float(b.get("timeoutS", 120))),
        ("POST", "/skill/stop"): lambda q, b: runtime.stop_motion(),
        ("POST", "/clock/sleep"): lambda q, b: {"t": runtime.link.sleep(float(b["seconds"]))},
        ("GET", "/ego/inputs"): lambda q, b: {"inputs": runtime.inputs},
    }
