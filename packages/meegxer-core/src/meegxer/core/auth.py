"""Who may talk to the daemon: a per-daemon token, checked by ZeroMQ (ZAP, PLAIN).

When the daemon starts it makes a random token, writes it to a file only the
user can read (``<config folder>/daemon/<control endpoint>.token``) and accepts
a connection on its control and display sockets only with that token as the
PLAIN password. Clients of the same user (GUI, CLI, scripts) read the file; the
``MEEGXER_TOKEN`` environment variable overrides it. The token keeps other
users and programs on the same computer out; PLAIN is not encrypted, so a
daemon on another computer still needs CurveZMQ (planned with the remote GUI).
See ``spec/control-api.md``.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
import threading
from pathlib import Path

import zmq

from meegxer.core.settings import config_dir

TOKEN_ENV = "MEEGXER_TOKEN"
USER = "meegxer"
ZAP_ENDPOINT = "inproc://zeromq.zap.01"  # fixed by ZeroMQ (RFC 27)


def token_path(control_endpoint: str) -> Path:
    name = re.sub(r"[^A-Za-z0-9]+", "_", control_endpoint).strip("_")
    return config_dir() / "daemon" / f"{name}.token"


def read_token(control_endpoint: str) -> str | None:
    """The token for the daemon at ``control_endpoint``, or None without one."""
    if os.environ.get(TOKEN_ENV):
        return os.environ[TOKEN_ENV]
    try:
        return token_path(control_endpoint).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_token(control_endpoint: str, token: str) -> Path:
    """Write the token readable by the user only (0600 in a 0700 folder; on
    Windows the config folder is already private to the account)."""
    path = token_path(control_endpoint)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    os.chmod(path, 0o600)  # an existing file keeps its mode through O_CREAT
    return path


def remove_token(control_endpoint: str, token: str) -> None:
    """Remove the token file if it is still this daemon's."""
    path = token_path(control_endpoint)
    try:
        if path.read_text(encoding="utf-8").strip() == token:
            path.unlink()
    except OSError:
        pass


def missing_token_hint(control_endpoint: str) -> str:
    return (
        f"no token for {control_endpoint} at {token_path(control_endpoint)}: "
        "is the daemon running under your account? (or set MEEGXER_TOKEN)"
    )


class Guard:
    """Server side: one token, checked by a ZAP handler (RFC 27) on the daemon's
    context for every socket that ``protect`` marked.

    A plain thread with a blocking poll: pyzmq's ThreadAuthenticator runs on
    zmq.asyncio, which on Windows' default (Proactor) event loop needs tornado
    and otherwise dies at once, leaving every login refused."""

    def __init__(self, context: zmq.Context[zmq.Socket[bytes]]) -> None:
        self.token = secrets.token_urlsafe(32)
        self._context = context
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="zap",
                                        daemon=True)  # fmt: skip
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("the daemon's token check did not start")

    @staticmethod
    def protect(socket: zmq.Socket[bytes]) -> None:
        """Before ``bind``: accept only clients with the token."""
        socket.plain_server = True

    def _allowed(self, mechanism: bytes, credentials: list[bytes]) -> bool:
        if mechanism != b"PLAIN" or len(credentials) != 2:
            return False
        user, password = credentials
        return user == USER.encode() and hmac.compare_digest(
            password, self.token.encode()
        )

    def _serve(self) -> None:
        socket = self._context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(ZAP_ENDPOINT)
        self._ready.set()
        try:
            while not self._stop.is_set():
                if not socket.poll(100):
                    continue
                # version, request id, domain, address, identity, mechanism, creds
                request = socket.recv_multipart()
                version, request_id = request[0], request[1]
                ok = len(request) >= 6 and self._allowed(request[5], request[6:])
                socket.send_multipart([
                    version, request_id, b"200" if ok else b"400",
                    b"OK" if ok else b"invalid token", USER.encode() if ok else b"",
                    b"",
                ])  # fmt: skip
        finally:
            socket.close(linger=0)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)


def login(socket: zmq.Socket[bytes], token: str | None) -> None:
    """Client side, before ``connect``: present the token (if there is one)."""
    if token:
        socket.plain_username = USER.encode()
        socket.plain_password = token.encode()
