"""Head plot of band power in the spectrum view."""

import math

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QWidget

from meegxer.gui.headplot import (
    HeadInput,
    HeadMap,
    Smoother,
    band_power,
    electrode_xy,
    layout,
    peak_frequency,
    render,
    spatial_colors,
    srgb_to_oklab,
    wheel_oklab,
)
from meegxer.gui.signal_view import SignalView, Window
from meegxer.gui.theme import load_theme

LABELS = ("Fp1", "Fp2", "F3", "F4", "Cz", "P3", "P4", "O1", "O2", "EOG")
RATE = 250.0


def dispose(qapp: QApplication, *widgets: QWidget) -> None:
    for w in widgets:
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def spectra(
    alpha: dict[str, float], f_alpha: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Flat 1 µV²/Hz spectra plus a Gaussian alpha peak per channel."""
    f = np.arange(0, 125.5, 0.5)
    peak = np.exp(-0.5 * ((f - f_alpha) / 0.8) ** 2)
    psd = np.array([1.0 + alpha.get(lb, 0.0) * peak for lb in LABELS])
    return f, psd


def window(probe: str, alpha: dict[str, float], seconds: float = 10.0) -> Window:
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(3)
    mid = np.array(
        [rng.standard_normal(len(t)) * 2e-6
         + alpha.get(lb, 0.0) * 1e-5 * np.sin(2 * np.pi * 10 * t) for lb in LABELS]
    )  # fmt: skip
    return Window(probe, LABELS, t, mid - 1e-8, mid + 1e-8,
                  np.zeros(len(t), dtype=bool), RATE)  # fmt: skip


OCCIPITAL = {"O1": 50.0, "O2": 50.0, "P3": 10.0, "P4": 10.0}


# -- numbers ------------------------------------------------------------------------


def test_band_power_is_the_integral_of_the_psd() -> None:
    f = np.arange(0, 50.5, 0.5)
    psd = np.full((2, f.size), 2.0)  # µV²/Hz
    assert band_power(f, psd, 8.0, 13.0) == pytest.approx([10.0, 10.0])
    # narrower than the resolution: the nearest bin times its width
    assert band_power(f, psd, 10.1, 10.2) == pytest.approx([1.0, 1.0])


def test_peak_frequency_needs_a_peak_inside_the_band() -> None:
    f, psd = spectra({"O1": 20.0}, f_alpha=10.5)
    rising = np.vstack([psd, f])  # a last row that only rises: its max is an edge
    peaks = peak_frequency(f, rising, 8.0, 13.0)
    assert peaks[LABELS.index("O1")] == pytest.approx(10.5)
    assert math.isnan(peaks[-1])


def test_smoothing_is_an_ema_with_a_time_constant() -> None:
    smoother = Smoother(tau=2.0)
    f = np.arange(3.0)
    first, second = np.zeros((1, 3)), np.ones((1, 3))
    assert smoother.update("tap", ("Cz",), f, first, now=0.0) is first
    got = smoother.update("tap", ("Cz",), f, second, now=2.0)  # one time constant
    np.testing.assert_allclose(got, 1 - math.exp(-1))
    # new channels start afresh; tau 0 is off
    assert smoother.update("tap", ("Pz",), f, second, now=3.0) is second
    smoother.tau = 0.0
    assert smoother.update("tap", ("Pz",), f, first, now=4.0) is first


def test_interpolation_is_linear_inside_and_voronoi_outside() -> None:
    points = np.array([[0.0, 0.6], [-0.6, -0.3], [0.6, -0.3], [0.0, 0.0]])
    lay = layout(points)
    plane = 2 * points[:, 0] - points[:, 1] + 1  # a linear field is reproduced
    got = lay.interpolate(plane)
    axis = np.linspace(-lay.extent, lay.extent, lay.inside.shape[0])
    gx, gy = np.meshgrid(axis, axis[::-1])
    px, py = gx[lay.inside], gy[lay.inside]
    hull = lay.weight[:, 1] != 0  # pixels inside a triangle
    np.testing.assert_allclose(got[hull], (2 * px - py + 1)[hull], atol=1e-9)
    # outside the triangulation: the nearest electrode's own value
    far = ~hull
    d = np.hypot(px[far, None] - points[:, 0], py[far, None] - points[:, 1])
    np.testing.assert_allclose(got[far], plane[np.argmin(d, axis=1)])


def test_spatial_blend_makes_equal_power_stand_out_equally() -> None:
    theme = load_theme("light")
    bg = srgb_to_oklab((251, 251, 250))
    wheel = wheel_oklab([90.0, 0.0, 200.0], [1.0, 0.0, 0.6], theme)  # rim, Cz, mid
    floor = float(np.min(np.linalg.norm(wheel - bg, axis=-1)))
    for s in (0.25, 0.5, 1.0):
        out = spatial_colors(wheel, np.full(3, s), bg, floor)
        np.testing.assert_allclose(np.linalg.norm(out - bg, axis=-1), s * floor)
    full = spatial_colors(wheel, np.ones(3), bg, floor)
    weakest = int(np.argmin(np.linalg.norm(wheel - bg, axis=-1)))
    np.testing.assert_allclose(full[weakest], wheel[weakest])  # full colour


def test_bipolar_channels_sit_between_their_electrodes() -> None:
    xy = electrode_xy(["O1-O2", "O1", "O2", "EOG"])
    assert "EOG" not in xy
    assert xy["O1-O2"] == pytest.approx(
        ((xy["O1"][0] + xy["O2"][0]) / 2, (xy["O1"][1] + xy["O2"][1]) / 2)
    )


# -- maps ---------------------------------------------------------------------------


def head(**kw: object) -> HeadMap | str:
    f, psd = spectra(OCCIPITAL)
    data = HeadInput(labels=LABELS, f=f, psd=psd, band=(8.0, 13.0))
    for k, v in kw.items():
        setattr(data, k, v)
    return render(data, load_theme("dark"))


def lightness(m: HeadMap, label: str) -> float:
    e = next(e for e in m.electrodes if e.label == label)
    return float(srgb_to_oklab(e.color)[0])


def test_every_scheme_maps_where_the_power_is() -> None:
    for scheme in ("spatial", "heat"):
        m = head(scheme=scheme)
        assert isinstance(m, HeadMap) and m.unplaced == ["EOG"]
        assert m.rgba.shape[2] == 4 and m.rgba[..., 3].any()
        # on the dark theme, more power is lighter (further from the background)
        assert lightness(m, "O1") > lightness(m, "Fp1") + 0.2
    m = head(relative=True)
    assert isinstance(m, HeadMap) and m.caption.startswith("Relative")


def test_change_compares_with_the_first_comparison_tap() -> None:
    assert head(scheme="change") == "Change needs a second tap (click a ◇ in the chain)"
    _, before = spectra({lb: 50.0 for lb in LABELS})
    m = head(scheme="change", compare_psd=before, compare_labels=LABELS)
    assert isinstance(m, HeadMap)
    text = {e.label: e.text for e in m.electrodes}
    assert text["Fp1"].startswith("Fp1: -") and "dB" in text["Fp1"]  # removed
    assert text["O1"] == "O1: +0.0 dB"  # unchanged where the alpha stayed


def test_peak_frequency_leaves_out_electrodes_without_a_peak() -> None:
    m = head(scheme="peak")
    assert isinstance(m, HeadMap)
    by = {e.label: e for e in m.electrodes}
    assert by["O1"].shown and by["O1"].text == "O1: 10.0 Hz"
    assert not by["Fp1"].shown  # flat spectrum: no peak inside the band
    assert by["Fp1"].text == "Fp1: no peak in the band"


def test_hidden_channels_are_left_out_of_the_map() -> None:
    m = head(hidden=frozenset({"O1"}))
    assert isinstance(m, HeadMap)
    o1 = next(e for e in m.electrodes if e.label == "O1")
    assert not o1.shown and o1.text.endswith("(hidden)")


def test_maps_explain_why_they_cannot_be_drawn() -> None:
    f, psd = spectra({})
    no_pos = HeadInput(labels=("A", "B", "C"), f=f, psd=psd[:3], band=(8.0, 13.0))
    assert "No electrode positions" in str(render(no_pos, load_theme("dark")))
    assert "beyond this view's 125 Hz" in str(head(band=(130.0, 140.0)))


# -- the spectrum view --------------------------------------------------------------


def view(qapp: QApplication) -> SignalView:
    v = SignalView(load_theme("dark"))
    v.resize(1100, 450)
    v.show()
    v._set_mode("spectrum")
    return v


def test_the_spectrum_view_shows_the_head_and_its_band(qapp: QApplication) -> None:
    v = view(qapp)
    main = window("main", OCCIPITAL)
    v.refresh(lambda p, s: main, "main", [])
    assert v.head.isVisible() and v.spectrum_bar.isVisible() and v.band.isVisible()
    assert isinstance(v.head.map, HeadMap)
    assert v.band_range() == (8.0, 13.0)

    v.band.setRegion((9.0, 12.0))  # dragged
    assert v.band_menu.currentText() == "Custom 9.0–12.0 Hz"  # noqa: RUF001
    v.band_menu.setCurrentIndex(1)
    v.band_menu.activated.emit(1)  # theta chosen
    assert v.band_range() == (4.0, 8.0)

    v.head.clicked.emit("O1")  # clicking an electrode hides its channel
    assert "O1" in v.channels.hidden
    v.head.clicked.emit("O1")
    assert "O1" not in v.channels.hidden

    v.head_toggle.setChecked(False)
    assert not v.head.isVisible()
    v.head_toggle.setChecked(True)
    v._set_mode("overlay")
    assert not (v.head.isVisible() or v.spectrum_bar.isVisible() or v.band.isVisible())
    dispose(qapp, v)


def test_live_spectra_are_smoothed_offline_ones_are_not(qapp: QApplication) -> None:
    v = view(qapp)
    quiet, loud = window("main", {}), window("main", {lb: 5.0 for lb in LABELS})
    v.smoothing.setCurrentIndex(4)  # 10 s
    v.panner.live = True
    first = v._spectra([quiet])[0][2]
    got = v._spectra([loud])[0][2]  # a moment later: barely moved
    v.panner.live = False
    offline = v._spectra([loud])[0][2]
    assert np.abs(got - first).max() < 0.05 * np.abs(offline - first).max()
    dispose(qapp, v)


def test_the_hovered_channel_is_drawn_bold(qapp: QApplication) -> None:
    v = view(qapp)
    main = window("main", OCCIPITAL)
    v.head.hovered.emit("O1")
    v.refresh(lambda p, s: main, "main", [])
    widths = [c.opts["pen"].widthF() for c in v.plot._curves if c.isVisible()]
    assert max(widths) == pytest.approx(4.0) and widths[-1] == pytest.approx(4.0)
    dispose(qapp, v)
