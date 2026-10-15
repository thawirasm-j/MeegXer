"""Mouse wheel over a number or choice field scrolls the panel, not the value.

Qt's default changes a spin box or combo box under the pointer on any wheel
turn, so scrolling the inspector or the chain silently edits parameters. With
the guard installed on the application, such a field takes wheel turns only
after it has been clicked (has focus); otherwise the turn goes to its parent.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QWidget


class WheelGuard(QObject):
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(obj, QAbstractSpinBox | QComboBox)
            and not obj.hasFocus()
        ):
            parent = obj.parentWidget()
            if isinstance(parent, QWidget):
                QApplication.sendEvent(parent, event)  # e.g. scroll the panel
            return True
        return False


def install_wheel_guard(app: QApplication) -> WheelGuard:
    """Install once per application; returns the guard (keep a reference)."""
    existing = app.findChild(WheelGuard)
    if existing is not None:
        return existing
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
