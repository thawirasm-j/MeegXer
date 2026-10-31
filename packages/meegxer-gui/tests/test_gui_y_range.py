"""The amplitude plots' y-range (visualization.y_range): zero-centred [-r, r]
or the data's [ymin, ymax], by itself and on a double click, with a y = 0
line either way; spectra and stacked channels keep their own fit."""

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from meegxer.gui.erp_view import Erp, ErpView
from meegxer.gui.signal_view import AmplitudeViewBox, StackedWindow, TracePlot
from meegxer.gui.theme import load_theme

x = np.linspace(0.0, 1.0, 100)
y = 1.0 + 4.0 * x  # 1 to 5 µV: all positive


def y_span(box: AmplitudeViewBox) -> tuple[float, float]:
    lo, hi = box.viewRange()[1]
    return float(lo), float(hi)


def fitted(plot: TracePlot, qapp: QApplication) -> tuple[float, float]:
    plot.box.enableAutoRange(axis=AmplitudeViewBox.YAxis)  # as a double click
    qapp.processEvents()
    return y_span(plot.box)


def test_zero_centred_or_data_range(qapp: QApplication) -> None:
    plot = TracePlot(load_theme("dark"))
    plot.resize(400, 300)
    plot.show()
    plot.draw([(x, y, pg.mkPen("w"))])  # cosmetic, as the views draw
    lo, hi = fitted(plot, qapp)
    assert lo == pytest.approx(-hi) and hi >= 5.0  # [-5, 5] and a margin
    assert plot.zero.isVisible()
    plot.box.set_y_range("data")
    lo, hi = fitted(plot, qapp)
    assert 0.5 < lo <= 1.0 and 5.0 <= hi < 5.5  # [1, 5] and a margin
    plot.box.set_y_range("zero_centered")
    plot.set_amplitude(False)  # a spectrum: its own fit, no zero line
    lo, hi = fitted(plot, qapp)
    assert lo > 0.5 and not plot.zero.isVisible()
    with pytest.raises(ValueError, match="y_range"):
        plot.box.set_y_range("wide")
    plot.close()
    plot.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_stacked_window_keeps_its_spacing(qapp: QApplication) -> None:
    stacked = StackedWindow(load_theme("dark"))
    assert not stacked.plot.box.amplitude and not stacked.plot.zero.isVisible()
    stacked.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_erp_view_is_zero_centred(qapp: QApplication) -> None:
    view = ErpView(load_theme("dark"))
    view.resize(600, 300)
    view.show()
    t = np.linspace(-0.2, 0.8, 101)
    values = (2e-6 + 6e-6 * np.clip(t, 0, None))[np.newaxis, np.newaxis]  # > 0
    view.refresh(Erp(("go",), (10,), ("Cz",), t, values), "ep:epochs", False)
    view.box.enableAutoRange(axis=AmplitudeViewBox.YAxis)
    qapp.processEvents()
    lo, hi = y_span(view.box)
    assert lo == pytest.approx(-hi) and hi >= 6.8  # µV, the peak 6.8
    view.box.set_y_range("data")
    qapp.processEvents()
    lo, _ = y_span(view.box)
    assert lo > 1.0  # the data start at 2 µV
    view.close()
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
