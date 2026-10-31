"""Undo and redo of the chain (M10.10): the edit history, and the window's
Undo and Redo through the daemon (offline edits, and a live run whose settings
go to the daemon without a restart)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QApplication

from meegxer.gui.history import EditHistory, describe_edit, edit_key


def chain(*nodes: tuple[str, str, dict[str, Any]]) -> str:
    return json.dumps({"nodes": [{"id": i, "node": n, "params": p}
                                 for i, n, p in nodes]}, sort_keys=True)  # fmt: skip


SRC: tuple[str, str, dict[str, Any]] = ("input", "meegxer.core.synthetic", {})


def hp(low: float) -> tuple[str, str, dict[str, Any]]:
    return ("hp", "meegxer.nodes.iir_filter", {"kind": "highpass", "low": low})


HP = hp(1.0)


def test_history_steps_merge_and_branch() -> None:
    h = EditHistory()
    h.reset(chain(SRC))
    assert h.undo() is None and h.redo() is None
    assert h.record(chain(SRC, HP), now=0.0)
    assert not h.record(chain(SRC, HP), now=0.1)  # nothing changed: no step
    # a value scrolled: quick changes of one setting are one step
    for t, low in ((10.0, 1.5), (10.5, 2.0), (11.0, 2.5)):
        h.record(chain(SRC, hp(low)), now=t)
    assert len(h.undo_stack) == 2
    assert h.undo() == chain(SRC, HP)  # back to low 1.0 in one step
    assert h.undo() == chain(SRC) and h.undo() is None
    assert h.redo() == chain(SRC, HP)
    h.record(chain(SRC, hp(3.0)), now=20.0)
    assert h.redo() is None  # a new edit drops what was undone
    # changes of one setting far apart are separate steps
    h.record(chain(SRC, hp(4.0)), now=30.0)
    assert h.undo() == chain(SRC, hp(3.0))


def test_history_keeps_at_most_its_limit() -> None:
    h = EditHistory(limit=3)
    h.reset(chain(SRC))
    for k in range(6):
        h.record(chain(SRC, hp(float(k))),
                 now=10.0 * k)  # fmt: skip
    assert len(h.undo_stack) == 3


def test_the_log_names_the_edit() -> None:
    two = chain(SRC, HP)
    assert edit_key(chain(SRC, HP), chain(SRC, hp(2.0))) == (
        "hp", "low")  # fmt: skip
    assert edit_key(chain(SRC), two) is None
    assert describe_edit(chain(SRC), two) == "added High-pass"
    assert describe_edit(two, chain(SRC)) == "removed High-pass"
    assert describe_edit(two, chain(SRC, hp(2.0))) == (
        "High-pass: Low cut-off")  # fmt: skip


def pump(qapp: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.02)


def test_undo_and_redo_in_the_window(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        start = [s.id for s in win.doc.strips]
        assert not win.undo_btn.isEnabled()  # a new window: nothing to undo
        win.insert_strip(len(win.doc.strips), "meegxer.nodes.psd")
        added = win.selected
        assert win.undo_btn.isEnabled()
        win.undo()
        assert [s.id for s in win.doc.strips] == start
        assert "Undo: added Power spectrum." in win.logbox.toPlainText()
        assert win.redo_btn.isEnabled()
        win.redo()
        assert [s.id for s in win.doc.strips] == [*start, added]
        assert "Redo: added Power spectrum." in win.logbox.toPlainText()
        # a setting changed twice in a row (a value stepped): one undo step
        low = win.doc.get("iirfilter1").params["low"]
        win.change_param("iirfilter1", "low", 2.0)
        win.change_param("iirfilter1", "low", 3.0)
        keys = {sc.key().toString(): sc for sc in win.findChildren(QShortcut)}
        keys["Ctrl+Z"].activated.emit()  # the shortcuts (offscreen: no key focus)
        assert win.doc.get("iirfilter1").params["low"] == low
        keys["Ctrl+Y"].activated.emit()
        assert win.doc.get("iirfilter1").params["low"] == 3.0
        keys["Ctrl+Z"].activated.emit()
        keys["Ctrl+Shift+Z"].activated.emit()
        assert win.doc.get("iirfilter1").params["low"] == 3.0
        assert win.modified()  # undo and redo are edits like any other
        # opening a pipeline starts a new history
        path = tmp_path / "p.json"
        path.write_text(json.dumps(win.doc.to_pipeline(taps=False)))
        win.open_file(str(path))
        assert not win.undo_btn.isEnabled() and not win.redo_btn.isEnabled()
    finally:
        win.close()


def test_undoing_a_setting_live_needs_no_restart(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    """Running live, an undo that only changes settings sends them to the
    daemon, as the edit did; one that changes the chain restarts the run."""
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.start()
        pump(qapp, 0.5)
        assert win.running
        low = win.doc.get("iirfilter1").params["low"]
        win.change_param("iirfilter1", "low", 2.0)
        win.undo()
        nodes = {n["id"]: n for n in win.call("status")["nodes"]}
        assert nodes["iirfilter1"]["params"]["low"] == low  # the daemon has it
        assert "Restarting" not in win.logbox.toPlainText()
        win.redo()
        nodes = {n["id"]: n for n in win.call("status")["nodes"]}
        assert nodes["iirfilter1"]["params"]["low"] == 2.0
    finally:
        win.stop()
        win.close()
