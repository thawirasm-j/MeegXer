"""BIDS-derivatives output from a GUI offline run."""

import subprocess
import time
from pathlib import Path

import mne
import mne_bids
import numpy as np
from PySide6.QtWidgets import QApplication


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_offline_run_writes_bids_derivatives(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
    raw = mne.io.RawArray(np.random.default_rng(0).standard_normal((3, 2500)) * 1e-5,
                          info, verbose="error")  # fmt: skip
    root = tmp_path / "ds"
    mne_bids.write_raw_bids(
        raw, mne_bids.BIDSPath(root=root, subject="01", task="rest", datatype="eeg"),
        format="EDF", allow_preload=True, verbose="error",
    )  # fmt: skip
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None and win.doc.mode == "offline"
        win.doc.source.params["path"] = str(
            root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf"
        )
        offered = [m["name"] for m in win.doc.nodes_for_palette()]
        assert "meegxer.core.bids_writer" in offered
        assert win.doc.incompatibility("meegxer.core.bids_writer") is None
        win.insert_strip(len(win.doc.strips), "meegxer.core.bids_writer")
        win.start()
        deadline = time.monotonic() + 60
        while win.offline_state != "done":
            assert win.offline_state != "error", win.logbox.toPlainText()
            assert time.monotonic() < deadline, "offline run did not finish"
            pump(qapp, 0.1)
        folder = root / "derivatives" / "meegxer" / "sub-01" / "eeg"
        assert (folder / "sub-01_task-rest_desc-clean_eeg.set").is_file()
        assert (folder / "sub-01_task-rest_desc-clean_meegxer.json").is_file()
    finally:
        win.close()
