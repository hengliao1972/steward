"""Extended experimental waits retain real elapsed time, sleeps and task success assertions."""

from pathlib import Path
from types import SimpleNamespace


def test_deadline_extension_changes_waits_without_changing_time_or_success_checks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from motion_matrix_extended_plugin import extend_functions
    sleeps, calls = [], []
    def call(url, payload, timeout):
        calls.append((payload, timeout))
        return "ok"
    namespace = {"time": SimpleNamespace(time=lambda: 10, sleep=sleeps.append), "call": call}
    source = '''
def _pump_until(timeout=120):
    return timeout
def test_task():
    time.sleep(0.25)
    deadline = time.time() + 300
    reply = call("url", {"timeoutS": 120}, timeout=130)
    assert reply == "ok"
    return deadline, _pump_until()
'''
    changes = extend_functions(source, {"test_task"}, 4, namespace, "test_deadline_input.py")
    assert namespace["test_task"]() == (1210, 480)
    assert sleeps == [0.25] and calls == [({"timeoutS": 480}, 520)]
    assert len(changes) == 4
