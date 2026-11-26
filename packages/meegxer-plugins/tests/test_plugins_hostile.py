"""The red team's attacks on the host, kept as regressions (docs/SECURITY.md):
a plugin is untrusted (rule 9), so what it sends back is bounded and checked
before the host keeps it. The ``hostile`` fixture plugin picks its attack by
``MODE``."""

from __future__ import annotations

import time
from collections.abc import Callable

from meegxer.plugins.worker import MAX_STDERR_LINE, LocalLauncher, PluginWorker


def test_a_stderr_line_without_end_is_cut(plugin: Callable[..., LocalLauncher]) -> None:
    """50 MB on stderr with no newline: the host keeps the first
    ``MAX_STDERR_LINE`` bytes of it (marked cut), then the lines after it."""
    worker = PluginWorker(plugin("hostile", MODE="stderr_flood"))
    try:
        worker.start()
        deadline = time.time() + 30
        while "after the flood" not in worker.stderr_tail() and time.time() < deadline:
            time.sleep(0.05)
        tail = worker.stderr_tail()
    finally:
        worker.close()
    lines = tail.splitlines()
    assert lines[-1] == "after the flood"
    assert lines[-2].endswith("… (cut)")
    assert len(lines[-2]) <= MAX_STDERR_LINE + len(" … (cut)")
    assert len(tail) < 2 * MAX_STDERR_LINE
