"""The robot agent's live thinking for the console: the reasoning blocks of the latest DSH
session under the robot's Harness home, read from the newest ``session.v4.jsonl.zstd`` file.

Reads are cached per file: a session file that did not change (same modification time and size)
is not decoded again, and each robot's read is limited to ``_TIME_BUDGET_S`` so a growing session
file cannot hold up the console's state refresh.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import threading
import time

_TEXT_LIMIT = 2000
_TIME_BUDGET_S = 0.5

_cache: dict[str, tuple[float, int, list[dict]]] = {}
_lock = threading.Lock()


def _session_file(robot_home: str) -> str | None:
    """The newest DSH session file of the robot's Harness home."""
    files = glob.glob(f"{robot_home}/sessions/*/*/session.v4.jsonl.zstd")
    return max(files, key=os.path.getmtime) if files else None


def _decode(path: str) -> list[dict]:
    out = subprocess.run(["zstd", "-dc", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"zstd failed to decode {path}: {out.stderr.strip()}")
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def _rows(path: str, deadline: float) -> list[dict]:
    """The session's rows, from the cache while the file is unchanged."""
    stat = os.stat(path)
    if time.monotonic() > deadline:
        raise TimeoutError(f"reading {path} would exceed the thinking budget")
    with _lock:
        cached = _cache.get(path)
        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
    rows = _decode(path)
    with _lock:
        _cache[path] = (stat.st_mtime, stat.st_size, rows)
        for other in [p for p in _cache if p != path][:20]:
            _cache.pop(other, None)
    return rows


def _thinking_of(rows: list[dict], limit: int) -> list[dict]:
    """One entry per step that has reasoning: its step number, whether it is the step still being
    thought out (after the latest ``step/start``), its text and its conclusion (the names of the
    tools it called), oldest first."""
    starts = [r for r in rows if r.get("type") == "step/start"]
    last_start = starts[-1]["time"] if starts else 0
    out = []
    for row in rows:
        if row.get("type") != "assistant/message":
            continue
        message = row["data"]["message"]
        reasoning = "".join(p["text"] for p in message["content"] if p["type"] == "reasoning").strip()
        if not reasoning:
            continue
        tools = [p["name"] for p in message["content"] if p["type"] == "tool-call"]
        out.append({"step": row["data"]["step"], "active": row["time"] >= last_start,
                    "text": reasoning[-_TEXT_LIMIT:], "conclusion": tools})
    return out[-limit:]


def thinking_state(robot_homes: dict[str, str], limit: int = 3) -> dict[str, dict]:
    """Live thinking per robot: its latest reasoning blocks and conclusions.

    :param robot_homes: robot id to its DSH home directory.
    :param limit: reasoning blocks per robot, newest last.
    :returns: per robot ``{entries, error}`` (one of them None)."""
    deadline = time.monotonic() + _TIME_BUDGET_S
    out: dict[str, dict] = {}
    for rid, home in robot_homes.items():
        path = _session_file(home)
        if path is None:
            continue
        try:
            out[rid] = {"entries": _thinking_of(_rows(path, deadline), limit), "error": None}
        except (OSError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
            out[rid] = {"entries": [], "error": f"{type(exc).__name__}: {exc}"}
    return out
