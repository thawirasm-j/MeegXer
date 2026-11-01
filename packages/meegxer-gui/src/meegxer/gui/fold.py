"""A row of buttons that folds what does not fit into its » button (Qt's
toolbar extension), so a narrow window, or text drawn wider (Windows, a
raised text size), never needs more width than the screen has."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QProxyStyle, QStyle, QToolBar, QWidget


class WideExtension(QProxyStyle):
    """The » button wide enough to show its arrow with the theme's button
    padding (Qt's default is 12 px)."""

    def pixelMetric(
        self, metric: QStyle.PixelMetric, option: Any = None, widget: Any = None
    ) -> int:
        if metric == QStyle.PixelMetric.PM_ToolBarExtensionExtent:
            return 26
        return super().pixelMetric(metric, option, widget)


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
    return bar
