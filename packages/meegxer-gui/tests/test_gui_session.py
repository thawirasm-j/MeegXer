"""Opening and saving sessions (spec/session.md) from the GUI."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.session import build_session, write_session

CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4",
            "O1", "O2"]  # fmt: skip


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def session_file(tmp_path: Path) -> Path:
    """A study folder with a recording and the session of an ICA run on it."""
    rng = np.random.default_rng(5)
    n = 250 * 40
    t = np.arange(n) / 250.0
    sources = np.vstack([np.sin(2 * np.pi * 10 * t), rng.laplace(size=n)])
    data = rng.standard_normal((len(CHANNELS), 2)) @ sources
    data = (data + 0.3 * rng.standard_normal((len(CHANNELS), n))) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(CHANNELS, 250.0, "eeg"),
                          verbose="error")  # fmt: skip
    raw.set_montage("colin27_1020")
    study = tmp_path / "study"
    study.mkdir()
    rec = study / "sub-01_raw.fif"
    raw.save(rec, verbose="error")
    pipeline = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(rec)}},
        {"id": "ica", "node": "meegxer.nodes.ica", "params": {"max_iter": 200}},
    ]}  # fmt: skip
    graph = Graph.from_dict(pipeline, Registry.from_entry_points())
    BatchScheduler().run(graph)
    return write_session(study / "sub-01.meegxer.json", build_session(graph))


def _window(control: str) -> Any:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    return MainWindow(DaemonLink(control))


def test_open_a_session_run_it_and_save_it_again(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    session_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = daemon
    win = _window(control)
    try:
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            lambda *a, **k: (str(session_file), ""))  # fmt: skip
        win.open_pipeline()
        assert [s.node for s in win.doc.strips][-1] == "meegxer.nodes.ica"
        assert win.doc.mode == "offline"
        log = win.logbox.toPlainText()
        assert "fitted models: ica" in log, log
        status = win.link.call("status")
        ica = next(n for n in status["nodes"] if n["id"] == "ica")
        assert ica["calibrated"] is True  # the session's model, in the daemon

        win.start()  # offline run with the session's model
        deadline = time.monotonic() + 60
        while win.offline_state != "done":
            assert win.offline_state != "error", win.logbox.toPlainText()
            assert time.monotonic() < deadline, "offline run did not finish"
            pump(qapp, 0.1)

        saved = session_file.with_name("again.meegxer.json")
        bare = str(saved.parent / "again")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (bare, ""))
        win.save_session()  # ".meegxer.json" is added to a bare name
        assert saved.is_file(), win.logbox.toPlainText()
        again = json.loads(saved.read_text())
        original = json.loads(session_file.read_text())
        assert again["name"] == "again"
        assert again["pipeline"]["nodes"][0]["params"]["path"] == "sub-01_raw.fif"
        assert (again["models"]["ica"]["content_hash"]
                == original["models"]["ica"]["content_hash"])  # fmt: skip
    finally:
        win.close()


def test_version_mismatch_asks_before_loading(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    session_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = json.loads(session_file.read_text())
    data["nodes"]["ica"]["version"] = "0.0.1"
    session_file.write_text(json.dumps(data))
    control, _ = daemon
    win = _window(control)
    try:
        asked: list[str] = []

        def answer(*args: Any, **kwargs: Any) -> QMessageBox.StandardButton:
            asked.append(str(args[2]))
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", answer)
        before = [s.node for s in win.doc.strips]
        win.open_session(str(session_file))
        assert asked and "recorded 0.0.1" in asked[0]
        assert [s.node for s in win.doc.strips] == before  # declined: unchanged

        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.StandardButton.Yes)  # fmt: skip
        win.open_session(str(session_file))
        assert [s.node for s in win.doc.strips][-1] == "meegxer.nodes.ica"
        ica = next(n for n in win.link.call("status")["nodes"] if n["id"] == "ica")
        assert ica["calibrated"] is False  # its model was dropped: calibrate again
    finally:
        win.close()
