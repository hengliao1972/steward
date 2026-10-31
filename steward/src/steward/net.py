"""Minimal JSON-over-HTTP server and client used by every Steward service in Phase 0.

Each service binds its own port, so the network topology expresses the information boundaries:
the World Control API port is reachable only inside the Steward plane, a robot port exposes only
that robot's sensor, actuator and telemetry channels, and the Ego API port is local to the robot.
"""

from __future__ import annotations

import errno
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Handler = Callable[[dict, Any], Any]


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Binary:
    """A non-JSON response body, optionally with extra response headers."""

    def __init__(self, data: bytes, content_type: str, headers: dict[str, str] | None = None):
        self.data = data
        self.content_type = content_type
        self.headers = headers or {}


NO_CONTENT = object()
"""A handler result answered with ``204 No Content`` (for example: no newer frame yet)."""


class JsonServer:
    """Serves ``routes`` on one port. A fixed port that is still held (a service just restarted on
    it) is retried for up to ``bind_wait_s`` seconds before the bind error is raised."""

    def __init__(self, routes: dict[tuple[str, str], Handler], host: str = "127.0.0.1", port: int = 0, name: str = "",
                 bind_wait_s: float = 0.0):
        self.routes = routes
        self.name = name
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D401 - silence default stderr logging
                return

            def _dispatch(self, method: str) -> None:
                parsed = urllib.parse.urlparse(self.path)
                query = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}
                handler = server.routes.get((method, parsed.path))
                if handler is None:
                    self._send(404, {"error": f"no route {method} {parsed.path}"})
                    return
                body = None
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    try:
                        body = json.loads(self.rfile.read(length))
                    except json.JSONDecodeError as exc:
                        self._send(400, {"error": f"invalid JSON body: {exc}"})
                        return
                try:
                    result = handler(query, body)
                except HttpError as exc:
                    self._send(exc.status, {"error": str(exc)})
                    return
                except Exception as exc:  # noqa: BLE001 - reported to the caller as a 500 body
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                if result is NO_CONTENT:
                    self.send_response(204)
                    self.end_headers()
                elif isinstance(result, Binary):
                    self.send_response(200)
                    self.send_header("Content-Type", result.content_type)
                    self.send_header("Content-Length", str(len(result.data)))
                    for key, value in result.headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(result.data)
                else:
                    self._send(200, result if result is not None else {})

            def _send(self, status: int, payload: Any) -> None:
                try:
                    data = json.dumps(payload, allow_nan=False).encode()
                except ValueError as exc:
                    # JSON has no NaN or Infinity: a result holding one is a server fault, reported as JSON.
                    status, data = 500, json.dumps({"error": f"result is not representable as JSON: {exc}"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - http.server naming
                self._dispatch("GET")

            def do_POST(self):  # noqa: N802 - http.server naming
                self._dispatch("POST")

        deadline, waited = time.monotonic() + bind_wait_s, False
        while True:
            try:
                self._httpd = ThreadingHTTPServer((host, port), _Handler)
                break
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE or port == 0 or time.monotonic() >= deadline:
                    raise
                waited = True
                time.sleep(0.5)
        if waited:
            print(f"{name or 'server'}: port {port} was still held; bound after waiting", file=sys.stderr, flush=True)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> JsonServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever, name=f"http-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread = None
        self._httpd.server_close()


def fetch(url: str, timeout: float = 10.0) -> tuple[int, bytes, dict[str, str]]:
    """GET a possibly binary resource: status, body and response headers (a 204 has an empty body).
    Raise :class:`HttpError` with the server message on an error status."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error", str(exc))
        except Exception:  # noqa: BLE001 - fall back to the HTTP status text
            message = str(exc)
        raise HttpError(exc.code, message) from None


def call(method: str, url: str, body: Any = None, timeout: float = 10.0) -> Any:
    """Send one JSON request; raise :class:`HttpError` with the server message on failure."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Type", "").startswith("application/json"):
                return json.loads(raw) if raw else {}
            return raw
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error", str(exc))
        except Exception:  # noqa: BLE001 - fall back to the HTTP status text
            message = str(exc)
        raise HttpError(exc.code, message) from None
