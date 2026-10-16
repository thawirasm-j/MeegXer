"""Plugin worker lifecycle: launch, handshake, request/reply, timeouts, restarts.

A worker is one long-running plugin process (normally a sandboxed container). The
host owns model and state, so a worker that crashes or hangs is simply killed and
restarted; the next request carries everything it needs (models are re-sent on
``unknown_model``).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

from meegxer.core.manifest import SPEC_VERSION
from meegxer.plugins.wire import Message, WireError, encode, read_message

UNSANDBOXED_ENV = "MEEGXER_ALLOW_UNSANDBOXED"


class PluginError(RuntimeError):
    """The plugin reported an error or broke the protocol."""


class PluginTimeout(PluginError):
    """The plugin did not answer in time and was killed."""


class PluginCrashed(PluginError):
    """The plugin process exited unexpectedly."""


def find_engine(preferred: str | None = None) -> str:
    """Return the container engine executable (docker or podman)."""
    for name in [preferred] if preferred else ["docker", "podman"]:
        if name and shutil.which(name):
            return name
    raise FileNotFoundError("no container engine found (install Docker or Podman)")


@dataclass(frozen=True)
class ContainerLauncher:
    """Runs a plugin image in a locked-down container (rule 9)."""

    image: str
    engine: str | None = None
    memory: str = "2g"
    cpus: float | None = None
    pids_limit: int = 256

    sandboxed = True

    def command(self, name: str) -> list[str]:
        cmd = [
            find_engine(self.engine),
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
        ]
        if self.cpus:
            cmd += ["--cpus", str(self.cpus)]
        return [*cmd, self.image]

    def kill(self, name: str) -> None:
        subprocess.run(
            [find_engine(self.engine), "kill", name],
            capture_output=True,
            check=False,
            timeout=30,
        )

    def environment(self) -> dict[str, str] | None:
        return None

    def describe(self) -> str:
        return f"container {self.image}"


@dataclass(frozen=True)
class LocalLauncher:
    """Runs a plugin as a local process, without a sandbox.

    For plugin development and tests only. Refused unless the environment variable
    ``MEEGXER_ALLOW_UNSANDBOXED=1`` is set or ``allow`` is passed explicitly.
    """

    argv: tuple[str, ...]
    allow: bool = False
    env: Mapping[str, str] = field(default_factory=dict)

    sandboxed = False

    def command(self, name: str) -> list[str]:
        if not (self.allow or os.environ.get(UNSANDBOXED_ENV) == "1"):
            raise PermissionError(
                "unsandboxed plugins are for development only; set "
                f"{UNSANDBOXED_ENV}=1 to allow {list(self.argv)}"
            )
        return list(self.argv)

    def kill(self, name: str) -> None:
        pass

    def environment(self) -> dict[str, str] | None:
        return {**os.environ, **self.env} if self.env else None

    def describe(self) -> str:
        return f"local process {' '.join(self.argv)} (UNSANDBOXED)"


type Launcher = ContainerLauncher | LocalLauncher

_live_workers: weakref.WeakSet[PluginWorker] = weakref.WeakSet()


@atexit.register
def _kill_all() -> None:
    for worker in list(_live_workers):
        worker.close(timeout=1.0)


class PluginWorker:
    """One plugin process with a synchronous request/reply API (thread-safe)."""

    def __init__(
        self,
        launcher: Launcher,
        *,
        timeout: float = 10.0,
        start_timeout: float = 120.0,
        max_restarts: int = 3,
        backoff: float = 0.5,
    ) -> None:
        self.launcher = launcher
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.max_restarts = max_restarts
        self.backoff = backoff
        self.restarts = 0  # total restarts, for diagnostics
        self._failures = 0  # consecutive failed attempts
        self.manifest: dict[str, Any] | None = None
        self.sdk = ""
        self.last_exit_code: int | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._name = ""
        self._replies: queue.Queue[Message | BaseException] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=200)
        self._next_id = 1
        self._loaded_models: set[str] = set()
        self._lock = threading.RLock()

    # -- lifecycle --------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self, lines: int = 20) -> str:
        return "\n".join(list(self._stderr)[-lines:])

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.alive and self.manifest is not None:
                return self.manifest
            self._spawn()
            reply = self._exchange(
                {"type": "hello", "spec_version": SPEC_VERSION}, {}, self.start_timeout
            )
            if reply.type != "hello" or not isinstance(
                reply.header.get("manifest"), dict
            ):
                raise PluginError(f"bad handshake reply: {reply.header}")
            self.manifest = reply.header["manifest"]
            self.sdk = str(reply.header.get("sdk", ""))
            return self.manifest

    def _spawn(self) -> None:
        self._name = f"meegxer-plugin-{uuid.uuid4().hex[:12]}"
        self._proc = subprocess.Popen(
            self.launcher.command(self._name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=self.launcher.environment(),
        )
        self._replies = queue.Queue()
        self._loaded_models.clear()
        proc = self._proc
        threading.Thread(target=self._read_replies, args=(proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()
        _live_workers.add(self)

    def _read_replies(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        while True:
            try:
                self._replies.put(read_message(proc.stdout))
            except BaseException as exc:  # EOF, protocol error
                self._replies.put(exc)
                return

    def _read_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, b""):
            self._stderr.append(line.decode(errors="replace").rstrip())

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        self.launcher.kill(self._name)
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        self.manifest = None

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self.alive:
                try:
                    self._exchange({"type": "shutdown"}, {}, timeout)
                    assert self._proc is not None
                    self.last_exit_code = self._proc.wait(timeout=timeout)
                except (PluginError, subprocess.TimeoutExpired, OSError):
                    pass
            self._kill()
            _live_workers.discard(self)

    # -- requests ----------------------------------------------------------------

    def _exchange(
        self,
        header: dict[str, Any],
        arrays: Mapping[str, npt.NDArray[Any]],
        timeout: float,
    ) -> Message:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise PluginCrashed(f"plugin is not running\n{self.stderr_tail()}")
        request_id = self._next_id
        self._next_id += 1
        try:
            proc.stdin.write(encode({**header, "id": request_id}, arrays))
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._kill()
            raise PluginCrashed(
                f"plugin closed its input\n{self.stderr_tail()}"
            ) from exc
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            message = f"no reply to {header['type']!r} within {timeout} s"
            raise PluginTimeout(f"{message}\n{self.stderr_tail()}") from None
        if isinstance(reply, BaseException):
            self._kill()
            if isinstance(reply, WireError):
                raise PluginError(f"protocol error: {reply}\n{self.stderr_tail()}")
            raise PluginCrashed(f"plugin exited\n{self.stderr_tail()}") from reply
        if reply.header.get("id") != request_id:
            self._kill()
            raise PluginError(f"reply id {reply.header.get('id')} != {request_id}")
        return reply

    def request(
        self,
        header: dict[str, Any],
        arrays: Mapping[str, npt.NDArray[Any]] | None = None,
        *,
        models: Mapping[str, tuple[Any, Mapping[str, npt.NDArray[Any]]]] | None = None,
    ) -> Message:
        """Send a request, restarting a dead or hung worker (with backoff).

        ``models`` maps model ids the request needs to their wire tree and arrays;
        they are loaded on demand (fresh workers, ``unknown_model``).
        """
        attempts = 0
        with self._lock:
            while True:
                try:
                    self.start()
                    for model_id, (tree, model_arrays) in (models or {}).items():
                        if model_id not in self._loaded_models:
                            self._load_model(model_id, tree, model_arrays)
                    reply = self._exchange(header, arrays or {}, self.timeout)
                    if (
                        reply.type == "error"
                        and reply.header.get("error") == "unknown_model"
                    ):
                        self._loaded_models.clear()
                        if attempts == 0 and models:
                            attempts += 1
                            continue
                    self._failures = 0
                    if reply.type == "error":
                        code = reply.header.get("error")
                        raise PluginError(f"{code}: {reply.header.get('message')}")
                    return reply
                except (PluginCrashed, PluginTimeout):
                    if self._failures >= self.max_restarts:
                        raise
                    self._failures += 1
                    self.restarts += 1
                    time.sleep(self.backoff * 2 ** (self._failures - 1))

    def _load_model(
        self, model_id: str, tree: Any, arrays: Mapping[str, npt.NDArray[Any]]
    ) -> None:
        reply = self._exchange(
            {"type": "load_model", "model_id": model_id, "model": tree},
            arrays,
            self.timeout,
        )
        if reply.type != "ok":
            raise PluginError(f"load_model failed: {reply.header}")
        self._loaded_models.add(model_id)

    def raw_exchange(self, header: dict[str, Any]) -> Message:
        """Send a request without retries (conformance checks)."""
        with self._lock:
            self.start()
            return self._exchange(header, {}, self.timeout)


def launcher_for(spec: Sequence[str] | str) -> Launcher:
    """``"image:tag"`` -> container; a list of argv -> local (unsandboxed)."""
    if isinstance(spec, str):
        return ContainerLauncher(spec)
    return LocalLauncher(tuple(spec))
