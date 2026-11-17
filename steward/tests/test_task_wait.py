"""Integration waits distinguish physical duration, slow execution and stalled wall time."""
import pytest

import test_cobot_ego as tasks


@pytest.mark.parametrize('outcome', ['succeeded', 'failed', 'simulation_timeout', 'wall_timeout'])
def test_task_wait_preserves_results_and_bounds_both_clocks(monkeypatch, outcome):
    wall = [0.]
    waits = [0]
    job = {'id': 'job-1', 'started': 10., 'state': 'running'}
    terminal = {**job, 'state': outcome, 'message': 'physical outcome', 'log': ['last motion']}

    def call(method, url, payload=None, **kwargs):
        if '/skill/status?' in url:
            return job
        if url.endswith('/ego/state'):
            return {'stamp': 131. if outcome == 'simulation_timeout' else 11.}
        assert url.endswith('/skill/wait')
        waits[0] += 1
        wall[0] += 211.  # One slow update must not hit the old 210-second wall limit.
        if waits[0] == 2 and outcome in ('succeeded', 'failed'):
            return terminal
        return job

    monkeypatch.setattr(tasks, 'call', call)
    monkeypatch.setattr(tasks.time, 'monotonic', lambda: wall[0])
    if outcome.endswith('_timeout'):
        with pytest.raises(AssertionError, match=outcome.split('_')[0] + ' duration exceeded'):
            tasks._wait_job('http://test', job)
    else:
        assert tasks._wait_job('http://test', job) == terminal
