"""Channels picked on a head plot; the standard 19 shown first above 19 EEG
channels (visualization.channels)."""

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from meegxer.gui.channel_picker import STANDARD_19, ChannelPicker, default_hidden
from meegxer.gui.signal_view import SignalView, Window
from meegxer.gui.theme import load_theme

# 32 channels of a 10-20 cap (old names T3/T4/T5/T6 for four of the 19)
CAP_32 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
          "FC6", "T3", "C3", "Cz", "C4", "T4", "CP5", "CP1", "CP2", "CP6", "T5",
          "P3", "Pz", "P4", "T6", "PO3", "PO4", "O1", "Oz", "O2", "VEOG",
          "ECG"]  # fmt: skip
TYPES = {"VEOG": "eog", "ECG": "ecg"}


def test_the_standard_19_first_above_19_eeg_channels() -> None:
    hidden = default_hidden(CAP_32, TYPES)
    shown = [lb for lb in CAP_32 if lb not in hidden]
    assert len(shown) == 19  # T3-T6 count as T7/T8/P7/P8
    assert {"VEOG", "ECG", "FC5", "Oz"} <= hidden
    # all EEG when asked, or when there are 19 or fewer
    assert default_hidden(CAP_32, TYPES, "all_eeg") == {"VEOG", "ECG"}
    few = ["Fz", "Cz", "Pz", "FC1", "EOG"]
    assert default_hidden(few, {"EOG": "eog"}) == {"EOG"}
    # none of the 19 by name (e.g. a 128-channel net): every EEG channel
    net = [f"E{k}" for k in range(1, 30)]
    assert default_hidden(net, {}) == frozenset()


def test_clicking_the_head_and_the_quick_choices(qapp: QApplication) -> None:
    picker = ChannelPicker(load_theme("dark"))
    labels = (*STANDARD_19, "FC5", "Aux1")
    seen: list[set[str]] = []
    picker.changed.connect(seen.append)
    picker.set_channels(labels, {"FC5"}, frozenset({"FC5"}))
    picker.resize(520, 300)
    picker.show()
    qapp.processEvents()
    assert set(picker.boxes) == {"Aux1"}  # no place on the head: a check box
    cz = picker.head._to_px(*picker.head.points["Cz"])
    QTest.mouseClick(picker.head, Qt.MouseButton.LeftButton,
                     pos=QPointF(cz).toPoint())  # fmt: skip
    assert seen[-1] == {"FC5", "Cz"}  # Cz hidden by its click
    picker.boxes["Aux1"].setChecked(False)
    assert "Aux1" in seen[-1]
    picker.show_standard()
    assert seen[-1] == {"FC5", "Aux1"}
    picker.show_all()
    assert seen[-1] == set()
    picker.show_default()
    assert seen[-1] == {"FC5"}
    picker.close()
    picker.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_signal_view_shows_the_19_first(qapp: QApplication) -> None:
    view = SignalView(load_theme("dark"))
    view.types = TYPES
    t = np.arange(500) / 50
    mid = np.random.default_rng(0).standard_normal((len(CAP_32), len(t))) * 1e-5
    win = Window("tap", tuple(CAP_32), t, mid - 1e-6, mid + 1e-6,
                 np.zeros(len(t), dtype=bool), 50.0)  # fmt: skip
    view.refresh(lambda p, s: win, "tap", [])
    assert len(CAP_32) - len(view.channels.hidden) == 19
    assert view.channels.text() == "Channels: 19/32"
    view.channels.toggle_channel("Oz")  # the head plot's click still works
    assert "Oz" not in view.channels.hidden
    assert "Oz" not in view.channels.picker.hidden  # the picker follows
    view.channel_rule = "all_eeg"
    view.refresh(lambda p, s: win, "tap", [])
    assert view.channels.hidden == {"VEOG", "ECG"}
    view.close()
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
