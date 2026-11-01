"""Clicking a strip selects it, which rebuilds the chain while Qt is still
delivering the click to the old strip widget: the old widgets must be deleted
later, not on the spot."""

import itertools
import subprocess
import time

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QMenu, QWidget


def test_clicking_a_strip_selects_it(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.chain import StripWidget
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            qapp.processEvents()
        strip = next(w for w in win.findChildren(StripWidget)
                     if w.strip_id == "iirfilter1")  # fmt: skip
        # before the fix: "Internal C++ object (StripWidget) already deleted"
        QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(20, 60))
        qapp.processEvents()
        assert win.selected == "iirfilter1"
    finally:
        win.close()


def _wheel(widget: QWidget, steps: int = 1) -> None:
    pos = QPointF(widget.rect().center())
    event = QWheelEvent(
        pos, widget.mapToGlobal(pos), QPoint(), QPoint(0, 120 * steps),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )  # fmt: skip
    QApplication.sendEvent(widget, event)


def test_wheel_changes_a_value_only_after_it_is_clicked(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            qapp.processEvents()
        win.select("iirfilter1")
        qapp.processEvents()
        spin = next(w for w in win.inspector.findChildren(QDoubleSpinBox)
                    if w.isVisible() and w.isEnabled())  # fmt: skip
        before = spin.value()
        _wheel(spin)  # pointer passes over the field while scrolling the panel
        assert spin.value() == before
        spin.setFocus()
        qapp.processEvents()
        if not spin.hasFocus():  # offscreen windows may not take focus
            win.activateWindow()
            spin.setFocus()
            qapp.processEvents()
        assert spin.hasFocus()
        _wheel(spin)
        assert spin.value() != before
    finally:
        win.close()


def test_no_widget_has_a_pixel_sized_font(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    """A font-size in px leaves pointSize() at -1, and Qt styles that copy the
    point size warn "QFont::setPointSize: Point size <= 0 (-1)" (seen on Windows)."""
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        qapp.processEvents()
        bad = [w for w in win.findChildren(QWidget) if w.font().pointSize() <= 0]
        assert not bad, [(type(w).__name__, w.styleSheet()) for w in bad]
    finally:
        win.close()


def test_the_add_strip_menu_has_categories_and_marks(qapp: QApplication) -> None:
    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import ChainView
    from meegxer.gui.pipeline_doc import PipelineDoc
    from meegxer.gui.theme import load_theme

    registry = Registry.from_entry_points()
    manifests = [manifest_to_dict(registry.get(n).manifest) for n in registry.names()]
    script = {**next(m for m in manifests if m["name"] == "meegxer.nodes.notch"),
              "name": "script.notch", "origin": "custom"}  # fmt: skip
    doc = PipelineDoc([*manifests, script])
    doc.set_source("meegxer.core.synthetic")
    view = ChainView()
    view.rebuild(doc, load_theme("dark"), statuses={}, taps={}, advanced=False,
                 selected=None)  # fmt: skip
    menu = view.build_palette(1)
    assert menu is not None
    headings = [a.text() for a in menu.actions()
                if not a.isEnabled() and not a.menu() and a.font().bold()]  # fmt: skip
    assert headings == ["● Preprocessing", "◆ Feature extraction", "▶ Output"]
    submenus = {a.text().replace("&&", "&"): a.menu() for a in menu.actions()
                if a.menu()}  # fmt: skip
    assert {"Filters", "Reference & channels", "Artifacts", "Spectral", "Files",
            "Streams"} <= set(submenus)  # fmt: skip

    def items(heading: str) -> list[str]:
        sub = submenus[heading]
        assert isinstance(sub, QMenu)
        return [a.text() for a in sub.actions()]

    filters = items("Filters")
    assert any("FIR" in t for t in filters)
    assert filters[-1].startswith("✎ ")  # the user's script, after the built-ins
    menu.deleteLater()
    view.deleteLater()


def test_a_rebuild_never_turns_the_old_page_into_a_window(qapp: QApplication) -> None:
    """Running or editing a value rebuilds the chain and the inspector. The old
    page waits for deferred deletion; left without a parent it would be a
    top-level window, which on Windows flashes up (v0.2.1a7)."""
    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import ChainView
    from meegxer.gui.inspector import Inspector
    from meegxer.gui.pipeline_doc import PipelineDoc
    from meegxer.gui.theme import load_theme

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.synthetic")
    hp = doc.insert(1, "meegxer.nodes.iir_filter")
    theme = load_theme("dark")
    view = ChainView()
    view.show()

    def rebuild() -> None:
        view.rebuild(doc, theme, statuses={}, taps={}, advanced=False, selected=None)

    rebuild()
    qapp.processEvents()
    old = view.widget()
    assert old is not None
    rebuild()  # the old page is now waiting to be deleted
    assert not old.isWindow() and not old.isVisible()
    inspector = Inspector()
    inspector.show()
    inspector.show_strip(doc, hp, advanced=False, running=False, node_status={})
    qapp.processEvents()
    page = inspector.widget()
    assert page is not None
    inspector.show_strip(doc, hp, advanced=False, running=False, node_status={})
    assert not page.isWindow() and not page.isVisible()
    for w in (view, inspector):
        w.close()
        w.deleteLater()
    qapp.processEvents()


def test_the_signal_is_grey_and_rail_ends_are_faint() -> None:
    """The signal stands apart from the ICs rail; past its last reader a rail
    runs on half transparent."""
    from meegxer.gui.chain import _toward_text, rail_colour, signal_colour
    from meegxer.gui.theme import load_theme

    for name in ("dark", "light"):
        theme = load_theme(name)
        assert signal_colour(theme) == theme.text_muted
        assert signal_colour(theme) != rail_colour(theme, "signal")
        assert _toward_text(rail_colour(theme, "features"), theme).alphaF() < 0.6


def test_strip_text_fits_the_strip(qapp: QApplication) -> None:
    """No text on a strip is clipped, whatever the platform's font: a long
    title takes two lines, the second ending in "…" if need be (measured in
    its own bold font), and a row whose name and value do not both fit puts
    the value on its own lines."""
    from PySide6.QtWidgets import QLabel

    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import StripWidget
    from meegxer.gui.pipeline_doc import PipelineDoc
    from meegxer.gui.theme import load_theme, stylesheet

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.synthetic")
    doc.insert(1, "meegxer.nodes.band_power")
    doc.insert(2, "meegxer.nodes.csp")
    doc.insert(3, "meegxer.nodes.ica")
    doc.input_mode = "offline"  # the ICA strip's "fits on the whole recording"
    theme = load_theme("dark")
    try:
        # larger text, as on Windows (wider fonts) or with Text size raised
        for scale, strip, advanced in itertools.product((1.0, 1.6), doc.strips,
                                                        (False, True)):  # fmt: skip
            qapp.setStyleSheet(stylesheet(theme, scale))
            card = StripWidget(doc, strip, theme, status="", advanced=advanced,
                               selected=True)  # fmt: skip
            card.show()
            qapp.processEvents()
            for label in card.findChildren(QLabel):
                if not label.text() or label.wordWrap():
                    continue
                lines = label.text().split("\n")
                assert len(lines) <= 2, (strip.id, label.text())
                drawn = max(label.fontMetrics().horizontalAdvance(t) for t in lines)
                assert drawn <= label.width(), (scale, strip.id, label.text())
            card.close()
            card.deleteLater()
    finally:
        qapp.setStyleSheet("")
    qapp.processEvents()


def test_strip_width_is_fixed_or_follows_the_text(qapp: QApplication) -> None:
    """appearance.strip_width: strips keep a fixed width (a percent of 158 px)
    whatever the text size, or with "text" grow with it, so a short name such
    as "Band power" is not shortened."""
    from PySide6.QtWidgets import QLabel

    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import STRIP_WIDTH, StripWidget
    from meegxer.gui.pipeline_doc import PipelineDoc
    from meegxer.gui.theme import load_theme, stylesheet

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.synthetic")
    strip = doc.insert(1, "meegxer.nodes.band_power")
    theme = load_theme("dark")

    def card(scale: float, size: int | str) -> tuple[int, str]:
        qapp.setStyleSheet(stylesheet(theme, scale))
        widget = StripWidget(doc, strip, theme, status="", advanced=False,
                             selected=False, size=size)  # fmt: skip
        widget.show()
        qapp.processEvents()
        title = next(w for w in widget.findChildren(QLabel)
                     if w.objectName() == "title")  # fmt: skip
        found = widget.width(), title.text()
        widget.close()
        widget.deleteLater()
        return found

    try:
        assert card(1.0, 100)[0] == card(1.6, 100)[0] == STRIP_WIDTH  # fixed
        assert card(1.0, 150)[0] == round(STRIP_WIDTH * 1.5)
        small, large = card(1.0, "text"), card(1.6, "text")
        assert large[0] > small[0] * 1.4 and large[1] == "Band power"
    finally:
        qapp.setStyleSheet("")
    qapp.processEvents()


def test_a_strip_adding_events_and_starting_a_rail_has_two_arrows(
    qapp: QApplication,
) -> None:
    """SSVEP detection (detections on) starts its own rail and adds to the
    Events rail: the two up-arrows stand side by side, the Events one not
    drawn over its own rail's start."""
    from PySide6.QtGui import QFontMetrics

    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import ChainView, RailBoard
    from meegxer.gui.pipeline_doc import PipelineDoc
    from meegxer.gui.theme import load_theme

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.file")
    doc.input_events = 24  # the recording has its cues: the input starts Events
    ssvep = doc.insert(1, "meegxer.nodes.ssvep_cca")
    ssvep.params["detections"] = True
    doc.insert(2, "meegxer.nodes.keyboard")
    view = ChainView()
    view.resize(1400, 300)
    view.show()
    view.rebuild(doc, load_theme("dark"), statuses={}, taps={}, advanced=False,
                 selected=None)  # fmt: skip
    qapp.processEvents()
    board = view.widget()
    assert isinstance(board, RailBoard)
    starts = {rail.id: x0 for rail, x0, _ in board.spans()}
    write = board._write_x(ssvep.id)
    assert write is not None and abs(write - starts[ssvep.id]) >= 10
    rail = next(r for r, _, _ in board.spans() if r.id == ssvep.id)
    name = board.label_box(rail, starts[ssvep.id], QFontMetrics(board.font_))
    assert name.right() < write  # no arrow runs through its name
    view.close()
    view.deleteLater()
    qapp.processEvents()


def test_long_text_takes_two_lines(qapp: QApplication) -> None:
    from PySide6.QtGui import QFont, QFontMetrics

    from meegxer.gui.chain import wrap_text

    fm = QFontMetrics(QFont())
    width = fm.horizontalAdvance("Common spatial") + 2
    text = wrap_text(fm, "Common spatial patterns (CSP)", width)
    first, second = text.split("\n")
    assert first == "Common spatial" and second.startswith("patterns")
    assert fm.horizontalAdvance(second) <= width
    three = wrap_text(fm, "delta:1-4,theta:4-8,alpha:8-13,beta:13-30", width)
    assert len(three.split("\n")) == 2 and three.endswith("…")  # the rest elided
    assert wrap_text(fm, "Notch", width) == "Notch"
    long_word = wrap_text(fm, "Supercalifragilistic", fm.horizontalAdvance("Super"))
    assert long_word.endswith("…")  # a word wider than the strip is cut


def test_the_input_card_says_what_it_needs_and_what_it_read(
    qapp: QApplication,
) -> None:
    from PySide6.QtWidgets import QLabel

    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import InputStrip
    from meegxer.gui.pipeline_doc import PipelineDoc
    from meegxer.gui.theme import load_theme

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.file")
    theme = load_theme("dark")

    def texts() -> list[str]:
        card = InputStrip(doc, doc.strips[0], theme, status="", advanced=False,
                          selected=False)  # fmt: skip
        out = [lb.text() for lb in card.findChildren(QLabel)]
        card.deleteLater()
        return out

    assert "choose the recording file on the Input strip" in texts()
    assert "not chosen" in texts()  # File: not chosen, not "-"
    doc.source.params["path"] = "/data/rec.edf"  # type: ignore[union-attr]
    doc.input_summary = "32 ch · 250 Hz · 2:00"
    shown = texts()
    assert "32 ch · 250 Hz · 2:00" in shown
    assert not any("choose the recording" in t for t in shown)
