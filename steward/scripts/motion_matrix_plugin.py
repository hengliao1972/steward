"""Isolated pytest instrumentation for the fixed physical-motion matrix.

All stages use the same main-thread host pump and worker test body. Existing assertions and
internal deadlines stay intact. World constructors receive only cumulative-odometry overrides.
"""

from dataclasses import asdict, replace
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import threading
import time

import pytest

TASKS = queue.Queue()
HOSTS = []
SIMS = []
RUNTIMES = []
OUTCOMES = []
ORIGINAL_PUMP = None


def _main(fn):
    if threading.current_thread() is threading.main_thread():
        return fn()
    done = threading.Event()
    result = {}
    TASKS.put((fn, done, result))
    if not done.wait(120):
        raise TimeoutError("matrix main-thread task did not complete")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _snapshot():
    directory = Path(os.environ["STEWARD_MATRIX_CASE_OUTPUT"])
    data = {"stage": os.environ["STEWARD_MATRIX_STAGE"], "maskSource": os.environ.get("STEWARD_MATRIX_MASK_SOURCE", "sam"),
            "simulations": [{"time": sim.time, "perception": asdict(sim.perception)} for sim in SIMS],
            "runtimes": [{"applied": getattr(runtime.config, "localization_apply", False),
                          "localization": getattr(runtime, "_localization_status", {}),
                          "integration": getattr(runtime, "_fusion_stats", {}),
                          "lastError": runtime.last_error} for runtime in RUNTIMES]}
    target = directory/"evidence.json"
    temporary = directory/"evidence.tmp"
    temporary.write_text(json.dumps(data, indent=2, default=str)+"\n", encoding="utf-8")
    temporary.replace(target)


def install_localization_options(monkeypatch, *, drift=None, localization_config=None,
                                 simulations=None, runtimes=None):
    """Configure test worlds and shared Ego runtimes for the duration of one test."""
    from steward.world.sim import WorldSim, PerceptionConfig
    from steward.ego.runtime import EgoRuntime, RuntimeConfig

    original_sim_init = WorldSim.__init__
    def sim_init(sim, *args, **kwargs):
        if drift is not None:
            cfg = kwargs.get("perception") or PerceptionConfig()
            kwargs["perception"] = cfg.with_odometry_drift(drift)
        original_sim_init(sim, *args, **kwargs)
        if simulations is not None:
            simulations.append(sim)
    if drift is not None or simulations is not None:
        monkeypatch.setattr(WorldSim, "__init__", sim_init)

    original_runtime_init = EgoRuntime.__init__
    def runtime_init(runtime, *args, **kwargs):
        if localization_config is not None:
            values = list(args)
            cfg = values[2] if len(values) > 2 else kwargs.get("config")
            cfg = replace(cfg or RuntimeConfig(), localization_config=localization_config,
                          localization_apply=True)
            if len(values) > 2:
                values[2] = cfg
            else:
                kwargs["config"] = cfg
            args = tuple(values)
        original_runtime_init(runtime, *args, **kwargs)
        if runtimes is not None:
            runtimes.append(runtime)
    if localization_config is not None or runtimes is not None:
        monkeypatch.setattr(EgoRuntime, "__init__", runtime_init)


@pytest.fixture(autouse=True)
def matrix_instrumentation(monkeypatch):
    from steward.world.sim import WorldSim
    from steward.world.server import StewardHost
    from steward.ego.runtime import EgoRuntime
    global ORIGINAL_PUMP
    stage = os.environ["STEWARD_MATRIX_STAGE"]
    drift = stage in ("drift", "drift-fused")
    fused = stage in ("drift-fused", "clean-fused")
    SIMS.clear()
    RUNTIMES.clear()
    HOSTS.clear()
    install_localization_options(monkeypatch, drift=drift, simulations=SIMS, runtimes=RUNTIMES,
                                 localization_config=os.environ["STEWARD_MATRIX_LOCALIZATION_CONFIG"] if fused else None)
    diagnostics = os.environ.get("STEWARD_MATRIX_SKILL_DIAGNOSTICS") == "1"
    if diagnostics:
        directory = Path(os.environ["STEWARD_MATRIX_CASE_OUTPUT"])
        (directory/"environment.json").write_text(json.dumps({
            "segmenterUrlPresent": "STEWARD_SEGMENTER_URL" in os.environ,
            "stage": stage, "maskSource": os.environ.get("STEWARD_MATRIX_MASK_SOURCE", "sam")
        }, indent=2)+"\n", encoding="utf-8")
        diagnostic_lock = threading.Lock()
        def instrument(name):
            original = getattr(EgoRuntime, name)
            def invoke(runtime, *args, **kwargs):
                record = {"method": name, "args": deepcopy(args), "kwargs": deepcopy(kwargs)}
                try:
                    response = original(runtime, *args, **kwargs)
                    record["response"] = deepcopy(response)
                    return response
                except Exception as error:
                    record["error"] = {"type": type(error).__name__, "message": str(error)}
                    raise
                finally:
                    with diagnostic_lock:
                        with (directory/"skill-events.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record, default=str)+"\n")
            monkeypatch.setattr(EgoRuntime, name, invoke)
        for name in ("predict", "run", "wait"):
            instrument(name)
    original_init, original_start, original_stop = StewardHost.__init__, StewardHost.start, StewardHost.stop
    ORIGINAL_PUMP = StewardHost.pump
    def init(host, *args, **kwargs):
        def create():
            original_init(host, *args, **kwargs)
            HOSTS.append(host)
        return _main(create)
    def stop(host):
        def close():
            original_stop(host)
            if host in HOSTS:
                HOSTS.remove(host)
        return _main(close)
    monkeypatch.setattr(StewardHost, "__init__", init)
    monkeypatch.setattr(StewardHost, "start", lambda host: _main(lambda: original_start(host)))
    monkeypatch.setattr(StewardHost, "stop", stop)
    monkeypatch.setattr(StewardHost, "pump", lambda host, timeout=0.05: _main(lambda: ORIGINAL_PUMP(host, timeout)))
    if os.environ.get("STEWARD_MATRIX_MASK_SOURCE", "sam") == "simulation":
        from motion_matrix_simulation_masks import install
        install(monkeypatch, _main)
    if os.environ.get("STEWARD_LOCALIZATION_EXPERIMENT") == "1":
        original_sim_stop = WorldSim.stop
        monkeypatch.setattr(WorldSim, "stop", lambda sim: _main(lambda: original_sim_stop(sim)))
    yield
    _snapshot()
    if diagnostics:
        # Job.to_dict truncates log history; the diagnostic archive retains every entry.
        jobs = [asdict(job) for runtime in RUNTIMES for job in runtime.executor.jobs.values()]
        (directory/"jobs.json").write_text(json.dumps(jobs, indent=2, default=str)+"\n", encoding="utf-8")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    arguments = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    done = threading.Event()
    result = {}
    def run():
        try:
            pyfuncitem.obj(**arguments)
        except BaseException as error:
            result["error"] = error
            result["traceback"] = error.__traceback__
        finally:
            done.set()
    worker = threading.Thread(target=run, name="matrix-test-body", daemon=True)
    worker.start()
    next_snapshot = time.monotonic()
    while not done.is_set():
        while True:
            try:
                fn, event, destination = TASKS.get_nowait()
            except queue.Empty:
                break
            try:
                destination["value"] = fn()
            except BaseException as error:
                destination["error"] = error
            finally:
                event.set()
        for host in list(HOSTS):
            ORIGINAL_PUMP(host, 0)
        if time.monotonic() >= next_snapshot:
            _snapshot()
            next_snapshot = time.monotonic()+2
        done.wait(0.001)
    worker.join()
    if "error" in result:
        raise result["error"].with_traceback(result["traceback"])
    return True


def pytest_runtest_logreport(report):
    OUTCOMES.append({"nodeid": report.nodeid, "phase": report.when, "outcome": report.outcome,
                     "duration": report.duration, "details": str(report.longrepr) if report.failed else None})


def pytest_sessionfinish(session, exitstatus):
    directory = Path(os.environ["STEWARD_MATRIX_CASE_OUTPUT"])
    (directory/"pytest-result.json").write_text(json.dumps({"exitstatus": int(exitstatus), "phases": OUTCOMES}, indent=2)+"\n",
                                                encoding="utf-8")
