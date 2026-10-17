"""Clicking a strip selects it, which rebuilds the chain while Qt is still
delivering the click to the old strip widget: the old widgets must be deleted
later, not on the spot."""

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
