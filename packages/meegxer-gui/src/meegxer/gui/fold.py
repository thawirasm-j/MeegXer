"""A row of buttons that folds what does not fit into its » button, so a
narrow window, or text drawn wider (Windows, a raised text size), never needs
more width than the screen has.

Qt's own » expands the toolbar in place, which works only for a toolbar
docked in a main window; ours sit in ordinary layouts, where that showed
nothing. So the » opens a menu of the folded items instead (``Overflow``):
a button becomes an item, a toggle a checkable item, a button with a menu
(Open's recent files, Save's session commands, Channels) and a choice list
(the time span) a submenu."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QMenu,
    QProxyStyle,
    QStyle,
    QToolBar,
    QToolButton,
    QWidget,
)

EXTENSION = "qt_toolbar_ext_button"  # the name Qt gives a toolbar's »


class WideExtension(QProxyStyle):
    """The » button wide enough to show its arrow with the theme's button
    padding (Qt's default is 12 px)."""

    def pixelMetric(
        self, metric: QStyle.PixelMetric, option: Any = None, widget: Any = None
    ) -> int:
        if metric == QStyle.PixelMetric.PM_ToolBarExtensionExtent:
            return 26
        return super().pixelMetric(metric, option, widget)


class Overflow(QObject):
    """Opens the menu of a toolbar's folded items when its » is clicked (or
    Space/Enter pressed on it), in place of Qt's in-place expansion."""

    def __init__(self, bar: QToolBar) -> None:
        super().__init__(bar)
        self.bar = bar
        self.menu: QMenu | None = None  # kept while it shows

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            self.show_menu()
            return True
        if kind in (QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):
            return True  # never Qt's expansion
        keys = (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if (kind == QEvent.Type.KeyPress and isinstance(event, QKeyEvent)
                and event.key() in keys):  # fmt: skip
            self.show_menu()
            return True
        return False

    def folded(self) -> list[QAction]:
        """The bar's actions whose items do not fit (hidden by its layout)."""
        out = []
        for action in self.bar.actions():
            widget = self.bar.widgetForAction(action)
            if action.isSeparator() or (widget is not None and not widget.isVisible()):
                out.append(action)
        # separators only between folded items
        while out and out[0].isSeparator():
            out.pop(0)
        while out and out[-1].isSeparator():
            out.pop()
        return out

    def build(self) -> QMenu:
        menu = QMenu(self.bar)
        for action in self.folded():
            if action.isSeparator():
                menu.addSeparator()
                continue
            widget = self.bar.widgetForAction(action)
            if widget is not None:
                _add(menu, widget)
        return menu

    def show_menu(self) -> QMenu:
        extension = self.bar.findChild(QToolButton, EXTENSION)
        self.menu = self.build()
        if extension is not None:
            self.menu.popup(extension.mapToGlobal(extension.rect().bottomLeft()))
        return self.menu


def _add(menu: QMenu, widget: QWidget) -> None:
    """``widget`` as a menu item: it does what clicking it would."""
    tip = widget.toolTip()
    if isinstance(widget, QComboBox):
        sub = menu.addMenu(tip.split("\n")[0] if tip else widget.currentText())
        for i in range(widget.count()):
            item = sub.addAction(widget.itemText(i))
            item.setCheckable(True)
            item.setChecked(i == widget.currentIndex())
            item.triggered.connect(lambda _=False, w=widget, n=i: _choose(w, n))
        return
    if not isinstance(widget, QAbstractButton):
        return
    text = widget.text().replace("&", "&&")
    if len(text.strip()) <= 2 and tip:  # ↶ ↷: named by their tooltip
        text = f"{text.strip()}  {tip.split(' (')[0].split(chr(10))[0]}"
    own_menu = widget.menu() if isinstance(widget, QToolButton) else None
    if own_menu is not None:
        popup = isinstance(widget, QToolButton) and (
            widget.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        if popup:  # Open…, Save…: the click, then what its arrow holds
            item = menu.addAction(text)
            item.triggered.connect(widget.click)
        if not own_menu.title():  # its own title ("Recent files"), else the text
            own_menu.setTitle(text)
        menu.addMenu(own_menu)
        return
    item = menu.addAction(text)
    if tip:
        item.setToolTip(tip)
    if widget.isCheckable():
        item.setCheckable(True)
        item.setChecked(widget.isChecked())
    item.setEnabled(widget.isEnabled())
    item.triggered.connect(widget.click)


def _choose(combo: QComboBox, index: int) -> None:
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def folding_bar(parent: QWidget | None = None) -> QToolBar:
    bar = QToolBar(parent)
    # owned by the bar, so Qt frees it after the bar (a style freed first
    # crashed Python's exit)
    extension = WideExtension()
    extension.setParent(bar)
    bar.setStyle(extension)
    bar.setMovable(False)
    bar.setFloatable(False)
    bar.setStyleSheet(
        "QToolBar { border: none; background: transparent; spacing: 6px; padding: 0; }"
    )
    button = bar.findChild(QToolButton, EXTENSION)
    if button is not None:
        overflow = Overflow(bar)
        button.installEventFilter(overflow)
        button.setToolTip("More: the buttons that do not fit")
    return bar


def overflow_of(bar: QToolBar) -> Overflow | None:
    """The bar's » menu maker (for tests and keyboard users)."""
    for child in bar.children():
        if isinstance(child, Overflow):
            return child
    return None
