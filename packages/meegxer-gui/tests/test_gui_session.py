"""Opening and saving sessions (spec/session.md) from the GUI."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from PySide6.QtCore import QTimer
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

        shared = session_file.with_name("shared.meegxer.json")  # Share session…
        monkeypatch.setattr(win, "keep_free_text", lambda text: False)  # remove it
        win.save_session(share=True, path=str(shared))
        text = shared.read_text()
        data = json.loads(text)
        assert "sub-01" not in text and "name" not in data
        assert data["pipeline"]["nodes"][0]["params"]["path"] == "<input>.fif"
        for key in ("data", "fingerprint", "content_hash"):  # the model itself
            assert data["models"]["ica"][key] == again["models"]["ica"][key]
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


def test_window_offers_to_recover_unfinished_recordings(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    import sys

    from meegxer.core.sinks import spool_paths

    def crash(path: Path) -> None:
        script = (
            "import os, numpy as np\n"
            "from meegxer.core.block import Block, SignalInfo\n"
            "from meegxer.core.sinks import SetSink\n"
            f"sink = SetSink({{'path': {str(path)!r}}})\n"
            "sink.write(Block(np.zeros((2, 200)), SignalInfo(('Fz', 'Cz'), 100.0),"
            " 0, 0.0))\n"
            "os._exit(1)\n"
        )
        subprocess.run([sys.executable, "-c", script], check=False)

    keep, drop = tmp_path / "keep.set", tmp_path / "drop.set"
    crash(keep)
    crash(drop)
    keep.write_bytes(b"a file already there")
    asked: list[dict[str, Any]] = []
    from meegxer.gui.main_window import MainWindow

    def choose(self: Any, rec: dict[str, Any]) -> str:
        asked.append(rec)
        return "recover" if rec["path"].endswith("keep.set") else "discard"

    monkeypatch.setattr(MainWindow, "recovery_choice", choose)
    control, _ = daemon
    window = _window(control)
    pump(qapp, 1.0)  # the offer comes once the window is up
    assert sorted(Path(r["path"]).name for r in asked) == ["drop.set", "keep.set"]
    assert keep.read_bytes() == b"a file already there"  # kept, not replaced
    recovered = tmp_path / "keep_recovered.set"
    assert mne.io.read_raw_eeglab(recovered, verbose="error").n_times == 200
    assert not any(p.exists() for p in [*spool_paths(keep), *spool_paths(drop)])

    monkeypatch.undo()  # the real question: its text, and "Ask me later"
    shown: list[str] = []

    def answer() -> None:
        box = next(w for w in qapp.topLevelWidgets()
                   if isinstance(w, QMessageBox) and w.isVisible())  # fmt: skip
        shown.append(box.text())
        next(b for b in box.buttons() if b.text() == "Ask me later").click()

    QTimer.singleShot(200, answer)
    rec = {**asked[0], "seconds": 125.0, "modified": "2026-10-19T03:44:00+00:00"}
    assert window.recovery_choice(rec) == "later"
    assert "2 channels, 2 min 5 s" in shown[0]
    window.close()
