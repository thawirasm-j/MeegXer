"""A one-line label that never sets its window's width: its text ends in
"…" where the layout gives it too little room, and then its tooltip holds
the whole (above a tooltip of its own, kept). ``text()`` returns the whole
text, as set."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    def __init__(
        self, text: str = "", parent: QWidget | None = None, keep: str = ""
    ) -> None:
        """``keep``: a text the label always has room for (the run's state),
        measured again whenever its font or style changes."""
        super().__init__(parent)
        self._full, self._keep, self._own_tip = "", keep, ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)
        self._keep_room()

    def minimumSizeHint(self) -> QSize:
        """Only the room it must keep: the shortened text's width would
        otherwise become its minimum, and it could never shrink again."""
        return QSize(self.minimumWidth(), super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        width = self.fontMetrics().horizontalAdvance(self._full) + 12
        return QSize(max(width, self.minimumWidth()), super().sizeHint().height())

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._keep_room()
            self._elide()

    def _keep_room(self) -> None:
        """Room for ``keep``: its text, the frame and the stylesheet's padding
        (known once laid out; 12 px until then)."""
        if not self._keep:
            return
        frame = self.width() - self.contentsRect().width() if self.width() else 0
        extra = max(frame, 12) + 2 * self.margin() + 6
        # and the "…" after it, when more text follows
        text = self.fontMetrics().horizontalAdvance(self._keep + " …")
        self.setMinimumWidth(text + extra)

    def setText(self, text: str) -> None:
        self._full = str(text)
        self._elide()

    def setToolTip(self, tip: str) -> None:
        self._own_tip = str(tip)
        self._show_tip()

    def _show_tip(self) -> None:
        cut = super().text() != self._full
        tip = "\n".join(t for t in ((self._full if cut else ""), self._own_tip) if t)
        super().setToolTip(tip)

    def text(self) -> str:
        return self._full

    def shown(self) -> str:
        """What is drawn: the text, or its start and "…"."""
        return super().text()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if event.oldSize().width() <= 0:  # first laid out: its padding is known
            self._keep_room()
        self._elide()

    def _elide(self) -> None:
        width = max(0, self.contentsRect().width())
        shown = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, width
        )
        if shown != super().text():
            super().setText(shown)
        self._show_tip()
