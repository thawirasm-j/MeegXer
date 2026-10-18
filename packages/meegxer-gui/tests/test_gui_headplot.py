"""Head plot of band power in the spectrum view."""

import math

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QWidget

from meegxer.gui.colors import channel_style
from meegxer.gui.headplot import (
    HeadInput,
    HeadMap,
    PeakHold,
    Smoother,
    aperiodic_fit,
    band_power,
    contour_segments,
    electrode_xy,
    flatten,
    layout,
    nice_step,
    peak_frequency,
    render,
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


def test_the_scale_top_rises_at_once_and_decays() -> None:
    hold = PeakHold(tau=2.0)
    assert hold.update("k", 10.0, now=0.0) == 10.0
    assert hold.update("k", 4.0, now=2.0) == pytest.approx(4 + 6 * math.exp(-1))
    assert hold.update("k", 12.0, now=2.5) == 12.0  # up at once
    assert hold.update("other band", 3.0, now=3.0) == 3.0  # starts again
    hold.tau = 0.0
    assert hold.update("other band", 1.0, now=4.0) == 1.0


def test_steps_are_round_numbers() -> None:
    assert [nice_step(x) for x in (0.3, 0.5, 1.6, 2.1, 7.0)] == [0.5, 0.5, 2, 5, 10]


def test_the_map_is_smooth_fills_the_head_and_never_overshoots() -> None:
    labels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
              "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]  # fmt: skip
    xy = electrode_xy(labels)
    points = np.array([xy[lb] for lb in labels])
    lay = layout(points)
    values = 3 * points[:, 0] - 2 * points[:, 1]
    grid = lay.interpolate(values)
    axis = np.linspace(-lay.extent, lay.extent, grid.shape[0])
    gx, gy = np.meshgrid(axis, axis[::-1])
    disc = np.hypot(gx, gy) <= lay.radius
    assert np.isfinite(grid[disc]).all()  # extrapolated to the edge
    assert grid[disc].min() >= values.min() and grid[disc].max() <= values.max()
    # exact at the electrodes (to within a pixel's worth of the slope)
    step = axis[1] - axis[0]
    for (x, y), v in zip(points, values, strict=True):
        r, c = round((lay.extent - y) / step), round((x + lay.extent) / step)
        assert grid[r, c] == pytest.approx(v, abs=4 * step)
    flat = lay.interpolate(np.full(len(points), 7.0))
    np.testing.assert_allclose(flat[disc], 7.0)


def test_contours_follow_the_level() -> None:
    n = 101
    rr, cc = np.indices((n, n), dtype=float)
    radius = np.hypot(rr - 50, cc - 50)
    seg = contour_segments(radius, 30.0)
    assert len(seg) > 100
    for r, c in ((seg[:, 0], seg[:, 1]), (seg[:, 2], seg[:, 3])):
        np.testing.assert_allclose(np.hypot(r - 50, c - 50), 30.0, atol=0.2)
    radius[:50] = np.nan  # no line through missing data
    assert (contour_segments(radius, 30.0)[:, [0, 2]] >= 50).all()


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
    hold = kw.pop("hold", None)
    for k, v in kw.items():
        setattr(data, k, v)
    return render(data, load_theme("dark"), hold=hold)  # type: ignore[arg-type]


def test_heat_maps_power_in_contour_bands_below_the_top() -> None:
    m = head()
    assert isinstance(m, HeadMap) and m.unplaced == ["EOG"]
    assert m.rgba[..., 3].any() and m.lines  # filled bands and their lines
    assert len(m.scale[2]) == 10 and m.scale[:2] == ("−20 dB", "max")  # noqa: RUF001
    assert m.caption == "Band power, 2 dB steps"
    narrow = head(db_range=10.0)
    assert isinstance(narrow, HeadMap) and len(narrow.scale[2]) == 5
    assert (narrow.rgba[..., 3] > 0).sum() < (m.rgba[..., 3] > 0).sum()
    # electrodes keep their channel colours whatever the map
    o1 = next(e for e in m.electrodes if e.label == "O1")
    assert o1.color == channel_style("O1", load_theme("dark")).color
    rel = head(relative=True)
    assert isinstance(rel, HeadMap) and rel.caption.startswith("Relative")


def test_a_held_top_shows_a_fading_map() -> None:
    seen: list[float] = []

    def hold(top: float) -> float:
        seen.append(top)
        return top + 6.0  # the scale still remembers a stronger moment

    now = head(db_range=10.0)
    held = head(db_range=10.0, hold=hold)
    assert isinstance(now, HeadMap) and isinstance(held, HeadMap)
    assert len(seen) == 1
    assert (held.rgba[..., 3] > 0).sum() < (now.rgba[..., 3] > 0).sum()


def test_change_compares_with_the_first_comparison_tap() -> None:
    assert head(scheme="change") == "Change needs a second tap (click a ◇ in the chain)"
    _, before = spectra({lb: 50.0 for lb in LABELS if lb not in ("O1", "O2")})
    m = head(scheme="change", compare_psd=before, compare_labels=LABELS)
    assert isinstance(m, HeadMap)
    text = {e.label: e.text for e in m.electrodes}
    assert text["Fp1"].startswith("Fp1: -") and "dB" in text["Fp1"]  # removed
    assert text["O1"].startswith("O1: +")  # added where the alpha grew
    assert any(bold for _, bold in m.lines)  # the 0 dB line between them


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
    assert not o1.shown and o1.text.endswith("(not EEG: off the map)")


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
    assert v.band_menu.currentText() == "Custom (9.0–12.0 Hz)"  # noqa: RUF001
    v.band_menu.setCurrentIndex(1)
    v.band_menu.activated.emit(1)  # theta chosen
    assert v.band_range() == (4.0, 8.0)

    v.head.clicked.emit("O1")  # clicking an electrode hides its channel's curve
    assert "O1" in v.channels.hidden
    v.refresh(lambda p, s: main, "main", [])  # the map keeps every EEG channel
    assert isinstance(v.head.map, HeadMap)
    assert next(e for e in v.head.map.electrodes if e.label == "O1").shown
    v.head.clicked.emit("O1")
    assert "O1" not in v.channels.hidden

    assert v.db_range.isEnabled()  # the range is heat's
    v.scheme.setCurrentIndex(1)
    assert not v.db_range.isEnabled()
    v.scheme.setCurrentIndex(0)

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


def test_live_the_heat_scale_top_is_held(qapp: QApplication) -> None:
    v = view(qapp)
    strong, weak = window("main", OCCIPITAL), window("main", {"O1": 5.0})
    v.panner.live = True
    v.refresh(lambda p, s: strong, "main", [])
    top = v.peak._held[1] if v.peak._held else None
    v.refresh(lambda p, s: weak, "main", [])  # a moment later: barely decayed
    assert top is not None and v.peak._held is not None
    assert v.peak._held[1] == pytest.approx(top, abs=0.5)
    dispose(qapp, v)


def test_the_hovered_channel_is_drawn_bold(qapp: QApplication) -> None:
    v = view(qapp)
    main = window("main", OCCIPITAL)
    v.head.hovered.emit("O1")
    v.refresh(lambda p, s: main, "main", [])
    widths = [c.opts["pen"].widthF() for c in v.plot._curves if c.isVisible()]
    assert max(widths) == pytest.approx(4.0) and widths[-1] == pytest.approx(4.0)
    dispose(qapp, v)


def test_offline_the_same_spectra_are_not_mapped_twice(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    import meegxer.gui.signal_view as sv

    calls: list[object] = []

    def counted(*args: object, **kw: object) -> object:
        calls.append(args)
        return render(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(sv, "render", counted)
    v = view(qapp)
    main = window("main", OCCIPITAL)
    v.refresh(lambda p, s: main, "main", [])
    v.refresh(lambda p, s: main, "main", [])  # the timer redraws: nothing changed
    assert len(calls) == 1
    v.refresh(lambda p, s: None, "main", [])  # a message in between...
    v.refresh(lambda p, s: main, "main", [])  # ...then the map is drawn again
    assert len(calls) == 2 and isinstance(v.head.map, HeadMap)
    v.band_menu.setCurrentIndex(1)
    v.band_menu.activated.emit(1)
    v.refresh(lambda p, s: main, "main", [])  # a new band is a new map
    assert len(calls) == 3
    dispose(qapp, v)


# -- aperiodic (1/f) removal --------------------------------------------------------


def one_over_f(
    exponent: float = 1.5, alpha: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A 1/f^exponent spectrum (10^-11 offset) with a little multiplicative
    noise and an optional alpha peak, as specparam models it (log10 power)."""
    rng = np.random.default_rng(seed)
    f = np.arange(0, 62.5, 0.5)
    with np.errstate(divide="ignore"):
        logp = -11 - exponent * np.log10(np.where(f > 0, f, np.nan))
    logp += alpha * np.exp(-0.5 * ((f - 10) / 1.0) ** 2)
    logp += 0.02 * rng.standard_normal(f.size)
    return f, 10 ** np.nan_to_num(logp, nan=0.0)[None, :]


def test_the_aperiodic_fit_is_specparams_robust_fit() -> None:
    specparam = pytest.importorskip("specparam")  # a dependency of meegxer-nodes

    for alpha in (0.0, 0.6):
        f, psd = one_over_f(alpha=alpha)
        offset, exponent = aperiodic_fit(f, psd)
        keep = (f >= 1) & (f <= 45)
        model = specparam.SpectralModel(verbose=False)
        robust = getattr(model.algorithm, "_robust_ap_fit", None)
        if robust is None:  # private in specparam; checked while it is there
            pytest.skip("specparam no longer exposes its robust aperiodic fit")
        ref = robust(f[keep], np.log10(psd[0, keep]))
        assert offset[0] == pytest.approx(ref[0], abs=1e-6)
        assert exponent[0] == pytest.approx(ref[1], abs=1e-6)
        assert exponent[0] == pytest.approx(1.5, abs=0.1)


def test_flattening_leaves_the_peak_above_zero() -> None:
    f, psd = one_over_f(alpha=0.6)
    flat = flatten(f, psd)
    assert np.isnan(flat[0, 0])  # 0 Hz
    assert flat[0, f == 10.0][0] == pytest.approx(6.0, abs=0.5)  # 0.6 log10 = 6 dB
    assert abs(np.median(flat[0, (f > 20) & (f < 45)])) < 0.5
    # rows are fitted independently; a row without data stays NaN
    both = np.vstack([psd, np.full_like(psd, np.nan)])
    _, exponent = aperiodic_fit(f, both)
    assert np.isfinite(exponent[0]) and np.isnan(exponent[1])


def test_heat_above_the_aperiodic_fit_starts_at_0_db() -> None:
    m = head(aperiodic=True)
    assert isinstance(m, HeadMap)
    assert m.scale[0] == "0 dB" and m.scale[1].startswith("+")
    assert m.caption.startswith("Band power above the aperiodic (1/f) fit")
    o1 = next(e for e in m.electrodes if e.label == "O1")
    assert "dB above the 1/f fit" in o1.text
    flat = head(aperiodic=True, psd=spectra({})[1])  # no peak: the floor of the scale
    assert isinstance(flat, HeadMap) and flat.scale[1] == "+3 dB"


def test_peak_frequency_above_the_fit_finds_a_peak_on_a_slope() -> None:
    f, psd = one_over_f(alpha=0.3)
    raw = peak_frequency(f, psd, 4.0, 13.0)
    assert np.isnan(raw[0])  # the 1/f slope peaks at the band's lower edge
    flat = 10 ** (flatten(f, psd) / 10)
    assert peak_frequency(f, np.nan_to_num(flat), 4.0, 13.0)[0] == 10.0


def test_the_view_can_remove_the_aperiodic_fit(qapp: QApplication) -> None:
    v = view(qapp)
    main = window("main", OCCIPITAL)
    v.refresh(lambda p, s: main, "main", [])
    assert "dB µV²/Hz" in v.plot.getPlotItem().getAxis("left").labelText
    v.aperiodic.setChecked(True)
    assert not v.measure.isEnabled()  # relative power means nothing above 1/f
    v.refresh(lambda p, s: main, "main", [])
    assert v.plot.getPlotItem().getAxis("left").labelText == "power above 1/f (dB)"
    assert isinstance(v.head.map, HeadMap) and v.head.map.scale[0] == "0 dB"
    assert v.view_state()["spectrum"]["aperiodic"] is True
    v.set_view_state({"spectrum": {"aperiodic": False}})
    assert not v.aperiodic.isChecked() and v.measure.isEnabled()
    dispose(qapp, v)


def test_the_map_ignores_the_menu_but_not_channel_types(qapp: QApplication) -> None:
    """The head plot maps every EEG channel with a position, whichever curves the
    channel menu shows; a non-EEG channel stays off it even when shown."""
    v = view(qapp)
    main = window("main", OCCIPITAL)
    v.types = {"Fp1": "eog"}
    v.refresh(lambda p, s: main, "main", [])
    for label in ("O1", "O2", "P3"):
        v.channels.toggle_channel(label)  # their curves hidden
    v.channels.toggle_channel("Fp1")  # the EOG-typed channel's curve shown
    assert {"O1", "O2", "P3"} <= v.channels.hidden and "Fp1" not in v.channels.hidden
    v.refresh(lambda p, s: main, "main", [])
    assert isinstance(v.head.map, HeadMap)
    by = {e.label: e for e in v.head.map.electrodes}
    assert by["O1"].shown and by["O2"].shown and by["P3"].shown
    assert not by["Fp1"].shown and by["Fp1"].text.endswith("off the map)")
    dispose(qapp, v)


def test_components_map_their_back_projected_power(qapp: QApplication) -> None:
    """Item 2: on a rail of components the head plot maps the selected
    components mixed back on the electrodes; all of them give the sensors."""
    sensors = window("main", OCCIPITAL)
    eeg = [i for i, lb in enumerate(LABELS) if lb != "EOG"]
    labels = tuple(LABELS[i] for i in eeg)
    x = (sensors.mins[eeg] + sensors.maxs[eeg]) / 2
    winv = np.random.default_rng(5).standard_normal((len(eeg), len(eeg)))
    s_ = np.linalg.solve(winv, x)  # activations: x = winv @ s
    names = tuple(f"IC{i + 1:02d}" for i in range(len(eeg)))
    ics = Window("ics", names, sensors.t, s_, s_, sensors.bypassed, RATE)
    direct = Window("main", labels, sensors.t, x, x, sensors.bypassed, RATE)

    v = view(qapp)
    v.smoothing.setCurrentIndex(0)  # off: each refresh maps its own spectra
    v.refresh(lambda p, s: direct, "main", [])
    assert isinstance(v.head.map, HeadMap)
    want = {e.label: e.text for e in v.head.map.electrodes}

    v.mixing = (labels, winv)
    v.refresh(lambda p, s: ics, "ics", [])
    m = v.head.map
    assert isinstance(m, HeadMap) and m.caption.endswith(
        f"{len(eeg)} of {len(eeg)} components, back-projected"
    )
    assert {e.label: e.text for e in m.electrodes} == want  # all ICs: the sensors

    for name in names[1:]:  # IC01 alone: its own scalp power
        v.channels.toggle_channel(name)
    v.refresh(lambda p, s: ics, "ics", [])
    m = v.head.map
    assert isinstance(m, HeadMap) and "1 of 9 components" in m.caption
    one = winv[:, :1] @ s_[:1]
    alone = Window("main", labels, sensors.t, one, one, sensors.bypassed, RATE)
    v.mixing = None
    v.refresh(lambda p, s: alone, "main", [])
    assert isinstance(v.head.map, HeadMap)
    assert {e.label: e.text for e in m.electrodes} == {
        e.label: e.text for e in v.head.map.electrodes
    }

    v.mixing = (labels, winv)
    v.refresh(lambda p, s: ics, "ics", [])  # the menu lists the ICs again
    for name in names:  # none selected
        v.channels.toggle_channel(name)
    v.refresh(lambda p, s: ics, "ics", [])
    assert v.head.map is None
    assert v.head.message == "Select components to map their power"
    dispose(qapp, v)


def test_the_1f_fit_leaves_out_what_the_filters_changed() -> None:
    """Item 6: the GUI's aperiodic fit skips the bands the stream's filters
    changed (below 1.5x a high-pass, around a notch), so the fit follows the
    true 1/f slope."""
    from meegxer.gui.headplot import aperiodic_fit

    f = np.arange(0.25, 60.0, 0.25)
    psd = (f**-2.0)[None]  # exponent 2
    psd = psd / (1 + (3.0 / f)[None] ** 8)  # a 3 Hz order-4 Butterworth high-pass
    psd[:, np.abs(f - 50.0) <= 1.0] *= 1e-3  # a notch's dip
    filters = (("highpass", 3.0, 0.0), ("notch", 50.0, 0.0))
    _, naive = aperiodic_fit(f, psd)
    _, careful = aperiodic_fit(f, psd, filters=filters)
    assert abs(careful[0] - 2.0) < 0.1
    assert abs(naive[0] - 2.0) > 0.3
