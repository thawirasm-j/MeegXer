"""Recording window, positions-based colours and the progressive offline view."""

import subprocess
import time
from pathlib import Path

import mne
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox

from meegxer.gui.colors import channel_style, wheel_positions
from meegxer.gui.theme import load_theme


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    names = ["Fp1", "Fp2", "Cz", "O1", "O2", "EOGx"]
    info = mne.create_info(names, 250.0, "eeg")
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((6, 250 * 60)) * 1e-5, info,
                          verbose="error")  # fmt: skip
    raw.set_annotations(mne.Annotations([5.0, 12.0, 30.0], [0.0, 0.0, 0.0],
                                        ["go", "stop", "go"]))  # fmt: skip
    path = tmp_path / "sub-01_raw.fif"
    raw.save(path, verbose="error")
    return path


def test_wheel_positions_follow_the_head() -> None:
    import mne as m

    montage = m.channels.make_standard_montage("colin27_1020")
    pos = {k: tuple(v) for k, v in montage.get_positions()["ch_pos"].items()}
    wheel = wheel_positions(pos)
    assert wheel["Cz"][1] < 0.15  # vertex near the centre
    assert abs(wheel["T8"][0] - 0) < 15 or abs(wheel["T8"][0] - 360) < 15  # right
    assert abs(wheel["Fpz"][0] - 90) < 15 and 0.8 < wheel["Fpz"][1] < 1.2
    dark = load_theme("dark")
    # a custom position overrides the table: Cz placed at the left rim turns blue
    moved = channel_style("Cz", dark, {"Cz": (180.0, 1.0)}).color
    assert moved[2] > moved[0]


def test_recording_window_edits_and_applies(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], recording: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(recording)
        win.open_recording()
        rw = win.recording_window
        assert rw.table.rowCount() == 6
        assert rw.labels.rowCount() == 2  # go, stop
        row = 5  # EOGx: rename, retype, mark bad
        label = rw.table.item(row, 1)
        assert label is not None
        label.setText("VEOG")
        combo = rw.table.cellWidget(row, 2)
        assert isinstance(combo, QComboBox)
        combo.setCurrentText("eog")
        bad = rw._bad_box(rw.table, row, 3)
        assert isinstance(bad, QCheckBox)
        bad.setChecked(True)
        rw.montage.setCurrentText("colin27_1020")
        drop = rw._bad_box(rw.labels, 1, 3)  # "stop"
        assert drop is not None
        drop.setChecked(True)
        rw.apply()
        rec = win.doc.source.recording
        assert rec["channels"]["EOGx"] == {"rename": "VEOG", "type": "eog", "bad": True}
        assert rec["montage"] == "colin27_1020"
        assert rec["events"]["drop"] == ["stop"]
        assert [e["label"] for e in rw.current["events"]] == ["go", "go"]
        positioned = [c for c in rw.current["channels"] if c["position"]]
        assert len(positioned) == 5  # every 10-20 label; VEOG has none
        assert set(win.view.wheel) == {"Fp1", "Fp2", "Cz", "O1", "O2"}
        assert win.doc.to_pipeline()["nodes"][0]["recording"] == rec
    finally:
        win.close()


def test_offline_view_scrolls_and_marks_events(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], recording: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None and win.doc.mode == "offline"
        win.doc.source.params["path"] = str(recording)
        win.start()
        deadline = time.monotonic() + 30
        while win.offline_state != "done":
            assert time.monotonic() < deadline, "offline run did not finish"
            pump(qapp, 0.1)
        main, _ = win._shown()
        assert main is not None
        first = win._offline_window(main, 10.0)
        assert first is not None and first.t[0] == pytest.approx(0.0, abs=0.01)
        win.timeline.bar.setValue(250 * 25)  # scroll to 25 s
        later = win._offline_window(main, 10.0)
        assert later is not None and later.t[0] == pytest.approx(25.0, abs=0.05)
        assert [e["label"] for e in win.view.events] == ["go", "stop", "go"]
        pump(qapp, 0.3)
        visible = [m for m in win.view.plot._marks if m.isVisible()]
        assert [m.value() for m in visible] == [30.0]  # only the event at 30 s
    finally:
        win.close()


def test_channel_lookup_fills_positions_by_name(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    names = ["EEG Fp1-REF", "EEG Cz-REF", "EEG T3-REF", "Status"]
    raw = mne.io.RawArray(np.zeros((4, 2500)), mne.create_info(names, 250.0, "eeg"),
                          verbose="error")  # fmt: skip
    path = tmp_path / "edf_style_raw.fif"
    raw.save(path, verbose="error")
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(path)
        win.open_recording()
        rw = win.recording_window
        assert all(c["position"] is None for c in rw.current["channels"])
        rw.lookup_source.setCurrentText("standard_1005")
        rw.look_up_positions()
        assert "Found 3 of 4" in rw.status.text() and "Status" in rw.status.text()
        rw.apply()
        rec = win.doc.source.recording
        assert set(rec["channels"]) == {"EEG Fp1-REF", "EEG Cz-REF", "EEG T3-REF"}
        assert all(len(e["position"]) == 3 for e in rec["channels"].values())
        positioned = {c["label"] for c in rw.current["channels"] if c["position"]}
        assert positioned == {"EEG Fp1-REF", "EEG Cz-REF", "EEG T3-REF"}
        # T3 (old name) lands on the left, where T7 is
        wheel = win.view.wheel
        assert 150 < wheel["EEG T3-REF"][0] < 210
    finally:
        win.close()


def test_non_eeg_channels_hidden_by_default_and_strips_can_include_them(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    info = mne.create_info(["Fz", "Cz", "VEOG", "ECG"], 250.0,
                           ["eeg", "eeg", "eog", "ecg"])  # fmt: skip
    raw = mne.io.RawArray(np.random.default_rng(0).standard_normal((4, 5000)) * 1e-5,
                          info, verbose="error")  # fmt: skip
    path = tmp_path / "types_raw.fif"
    raw.save(path, verbose="error")
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(path)
        win.start()
        deadline = time.monotonic() + 30
        while win.offline_state != "done":
            assert time.monotonic() < deadline
            pump(qapp, 0.1)
        pump(qapp, 0.3)
        assert win.view.channels.hidden == {"VEOG", "ECG"}
        assert win.view.channels.text() == "Channels: 2/4"
        win.change_channel_types("iirfilter1", ["eeg", "eog"])
        nodes = {n["id"]: n for n in win.doc.to_pipeline()["nodes"]}
        assert nodes["iirfilter1"]["channel_types"] == ["eeg", "eog"]
        assert "channel_types" not in nodes["notch1"]
    finally:
        win.close()


def test_offline_taps_at_another_rate_show_the_same_time(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from typing import Any

    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    asked: dict[str, int] = {}

    def call(method: str, **kw: Any) -> dict[str, Any]:
        asked[kw["probe"]] = kw["start"]
        return {"ch_labels": ["Cz"], "mins": [0.0], "maxs": [0.0],
                "bin_size": 1, "start_bin": kw["start"]}  # fmt: skip

    try:
        win.offline_sfreq = 1000.0  # the input's rate: the scrollbar's samples
        win.offline_probes = {
            p: {"sfreq": rate, "first_sample": 0, "n_filled": 10, "n_samples": 10}
            for p, rate in [("tap_in", 1000.0), ("tap_rs", 250.0)]
        }
        win.call = call  # type: ignore[method-assign]
        win._offline_window("tap_in", 1.0, offset=4000, rate=1000.0)
        win._offline_window("tap_rs", 1.0, offset=4000, rate=1000.0)
        assert asked == {"tap_in": 4000, "tap_rs": 1000}  # both 4 s in
    finally:
        win.close()


def test_a_chosen_recording_is_shown_at_once(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], recording: Path
) -> None:
    """Offline, choosing the file shows its signal and events before any run:
    the input is read and drawn on its tap; Run offline runs the chain."""
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        win.change_param("input", "path", str(recording))
        deadline = time.monotonic() + 30
        while win.offline_state != "done":
            assert time.monotonic() < deadline, "the preview did not finish"
            pump(qapp, 0.1)
        assert list(win.offline_probes) == ["tap_input"]
        assert win.taps["input"] == "main"
        main, _ = win._shown()
        assert main is not None
        shown = win._offline_window(main, 10.0)
        assert shown is not None and shown.t[0] == pytest.approx(0.0, abs=0.01)
        assert [e["label"] for e in win.view.events] == ["go", "stop", "go"]
        assert "The recording, as read" in win.logbox.toPlainText()
        win.start()  # the chain now runs as usual
        pump(qapp, 0.3)
        deadline = time.monotonic() + 30
        while win.offline_state != "done":
            assert time.monotonic() < deadline, "the run did not finish"
            pump(qapp, 0.1)
        assert "Offline run done" in win.logbox.toPlainText()
        assert len(win.offline_probes) > 1  # the chain's taps too
    finally:
        win.close()


def test_an_epochs_file_opens_on_its_erp(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    """An epochs file (-epo.fif) starts the epochs rail at the input: the
    preview shows its ERP per label, and its labels fill the class pickers."""
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow
    from meegxer.gui.pipeline_doc import tap_id

    rng = np.random.default_rng(0)
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    events = np.column_stack([np.arange(5) * 100, np.zeros(5, int),
                              [1, 2, 1, 2, 1]])  # fmt: skip
    path = tmp_path / "rec-epo.fif"
    mne.EpochsArray(rng.standard_normal((5, 3, 61)) * 1e-6, info, events,
                    tmin=-0.1, event_id={"target": 1, "standard": 2},
                    verbose="error").save(path, verbose="error")  # fmt: skip
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        win.change_param("input", "path", str(path))
        assert win.doc.input_rails == ("epochs",)
        assert win.doc.event_labels == ["target", "standard"]
        assert win.view_tap == "input:epochs"
        deadline = time.monotonic() + 30
        while win.offline_state != "done":
            assert time.monotonic() < deadline, "the preview did not finish"
            pump(qapp, 0.1)
        pump(qapp, 0.3)
        assert win.views.currentIndex() == 2  # the ERP view
        erp = win._offline_erp(tap_id("input:epochs"))
        assert erp is not None
        assert dict(zip(erp.conditions, erp.counts, strict=True)) == {
            "target": 3, "standard": 2}  # fmt: skip
        assert erp.channels == ("Fz", "Cz", "Pz")
    finally:
        win.close()


def test_errors_read_as_words() -> None:
    from meegxer.gui.main_window import plain_error

    assert plain_error("FileNotFoundError: input file not found: x.edf") == (
        "input file not found: x.edf"
    )
    trace = ('Traceback (most recent call last):\n  File "lsl.py", line 163\n'
             "TimeoutError: no LSL stream named 'Amp' found\n")  # fmt: skip
    assert plain_error(trace) == "no LSL stream named 'Amp' found"
    assert plain_error("RuntimeError: ValueError: bad") == "bad"
    assert plain_error("filter: Offline only") == "filter: Offline only"


def test_an_input_without_its_file_says_what_to_do(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], recording: Path
) -> None:
    """Choosing *Recording file* reads nothing yet (no error); the view and
    Run offline say to choose the file; a new pipeline starts clean."""
    from meegxer.core.templates import template_pipeline
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        if win.doc.mode != "offline":
            win.toggle_mode()
        assert "rror" not in win.logbox.toPlainText()  # nothing read, no error
        assert win._empty_text().endswith("choose the recording file on the "
                                          "Input strip")  # fmt: skip
        win.start()
        assert (
            win.logbox.toPlainText()
            .splitlines()[-1]
            .endswith("Cannot start: choose the recording file on the Input strip")
        )
        win.change_param("input", "path", str(recording.with_name("gone.fif")))
        assert win.logbox.toPlainText().splitlines()[-1].endswith(
            "Could not read the input: input file not found: "
            f"{recording.with_name('gone.fif')}")  # fmt: skip
        win.change_param("input", "path", str(recording))
        assert win._empty_text() == "Reading the recording…"
        deadline = time.monotonic() + 30
        while win.offline_state != "done":
            assert time.monotonic() < deadline, "the preview did not finish"
            pump(qapp, 0.1)
        assert win.offline_probes and win.view.events
        win._load_template("ERP study", template_pipeline("erp"))
        pump(qapp, 1.0)  # status polls: the preview's results stay gone
        assert win.offline_state == "" and not win.offline_probes
        assert win.view.events == []
        assert "choose the BIDS dataset folder" in win._empty_text()
    finally:
        win.close()
