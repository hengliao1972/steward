"""JSON-over-HTTP services: binding a fixed port that a stopped service still holds, and results
that JSON cannot represent."""

from __future__ import annotations

import json
import math
import socket
import threading
import urllib.error
import urllib.request

import pytest

from steward.net import JsonServer


def _held_port() -> socket.socket:
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    return held


def test_a_fixed_port_freed_within_the_wait_is_bound():
    held = _held_port()
    port = held.getsockname()[1]
    threading.Timer(0.6, held.close).start()
    server = JsonServer({}, port=port, name="probe", bind_wait_s=5.0).start()
    try:
        assert server.port == port
    finally:
        server.stop()


def test_a_fixed_port_still_held_after_the_wait_fails_loudly():
    held = _held_port()
    try:
        with pytest.raises(OSError):
            JsonServer({}, port=held.getsockname()[1], name="probe", bind_wait_s=0.6)
    finally:
        held.close()


def test_a_result_holding_a_non_finite_number_is_a_json_server_error():
    server = JsonServer({("GET", "/far"): lambda q, b: {"clearance": math.inf}}, port=0, name="probe").start()
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{server.url}/far", timeout=5)
        assert caught.value.code == 500
        assert "not representable as JSON" in json.loads(caught.value.read())["error"]
    finally:
        server.stop()
