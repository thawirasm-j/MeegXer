"""FIR filter, bad-channel interpolation, CSD and EOG regression."""

import mne
import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.registry import Registry
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.nodes.artifacts import EOGRegression, split_channels
from meegxer.nodes.filters import FIRFilter, FIRFilterZeroPhase
from meegxer.nodes.spatial import CSD, InterpolateBads, fit_sphere

SFREQ = 250.0
EEG = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
       "P7", "P3", "Pz", "P4", "P8", "O1", "O2")  # fmt: skip
POS = mne.channels.make_standard_montage("colin27_1020").get_positions()["ch_pos"]


def info(bads: tuple[str, ...] = (), eog: bool = True) -> SignalInfo:
    labels = (*EEG, "EOG") if eog else EEG
    return SignalInfo(
        labels, SFREQ, bads=bads,
        ch_types=tuple("eog" if lb == "EOG" else "eeg" for lb in labels),
        positions=tuple(None if lb == "EOG" else tuple(POS[lb]) for lb in labels),
    )  # fmt: skip


def block(data: np.ndarray, i: SignalInfo) -> Block:
    return Block(data, i, 0, 0.0)


def run(proc: object, b: Block, **params: object) -> Block:
    stage = Stage(proc, params)  # type: ignore[arg-type]
    out = stage.transform(b, signal_fingerprint(b.info)).block
    assert isinstance(out, Block)
    return out


def sines(*freqs: float, n_ch: int = 4, seconds: float = 8.0) -> np.ndarray:
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    return np.tile(sum(np.sin(2 * np.pi * f * t) for f in freqs), (n_ch, 1)) * 1e-5


# -- FIR ------------------------------------------------------------------------------


def test_fir_passes_the_band_and_removes_the_rest() -> None:
    x = sines(2.0, 10.0, 60.0)
    b = Block(x, SignalInfo(("A", "B", "C", "D"), SFREQ), 0, 0.0)
    for proc in (FIRFilter(), FIRFilterZeroPhase()):
        params = {"kind": "bandpass", "low": 6.0, "high": 20.0}
        if isinstance(proc, FIRFilter):
            params["phase"] = "causal"
        y = run(proc, b, **params).data[0, 750:-250]  # past the start-up
        spectrum = np.abs(np.fft.rfft(y))
        f = np.fft.rfftfreq(len(y), 1 / SFREQ)
        power = {hz: spectrum[np.argmin(np.abs(f - hz))] for hz in (2, 10, 60)}
        assert power[10] > 50 * power[2] and power[10] > 50 * power[60]


def test_zero_phase_fir_keeps_the_timing_and_matches_mne() -> None:
    x = sines(10.0)
    b = Block(x, SignalInfo(("A", "B", "C", "D"), SFREQ), 0, 0.0)
    y = run(FIRFilterZeroPhase(), b, kind="lowpass", low=1.0, high=30.0).data
    expected = mne.filter.filter_data(x, SFREQ, None, 30.0, phase="zero",
                                      fir_design="firwin", verbose="error")  # fmt: skip
    np.testing.assert_allclose(y, expected)
    mid = slice(500, -500)
    lag = np.argmax(np.correlate(y[0, mid], x[0, mid], "full")) - (len(x[0, mid]) - 1)
    assert lag == 0  # no delay


def test_fir_phase_auto_is_causal_live_and_zero_phase_offline() -> None:
    fir = FIRFilter()
    assert fir.for_mode({"phase": "auto"}, "online") is fir
    assert isinstance(fir.for_mode({"phase": "auto"}, "offline"), FIRFilterZeroPhase)
    with pytest.raises(ValueError, match="whole recording"):
        fir.for_mode({"phase": "zero"}, "online")


def test_causal_fir_is_the_same_block_by_block() -> None:
    rng = np.random.default_rng(1)
    b = Block(rng.standard_normal((4, 2000)) * 1e-5,
              SignalInfo(("A", "B", "C", "D"), SFREQ), 0, 0.0)  # fmt: skip
    params = {"kind": "highpass", "low": 1.0, "phase": "causal"}
    check_blockwise_equivalence(FIRFilter(), b, params, trials=4)


# -- interpolation and CSD -----------------------------------------------------------


def eeg_data(n: int = 1500, seed: int = 2) -> np.ndarray:
    """A smooth field over the scalp plus a little noise, and an EOG channel."""
    rng = np.random.default_rng(seed)
    xyz = np.array([POS[lb] for lb in EEG])
    sources = rng.standard_normal((3, n))
    eeg = (xyz @ sources) * 1e-4 + rng.standard_normal((len(EEG), n)) * 1e-7
    return np.vstack([eeg, rng.standard_normal((1, n)) * 1e-5])


def mne_raw(data: np.ndarray, bads: tuple[str, ...] = ()) -> mne.io.RawArray:
    raw = mne.io.RawArray(data[: len(EEG)], mne.create_info(list(EEG), SFREQ, "eeg"),
                          verbose="error")  # fmt: skip
    raw.set_montage(mne.channels.make_dig_montage(
        {lb: POS[lb] for lb in EEG}, coord_frame="head"), verbose="error")  # fmt: skip
    raw.info["bads"] = list(bads)
    return raw


def test_interpolation_matches_mne_and_leaves_the_rest() -> None:
    data = eeg_data()
    bads = ("C3", "Pz")
    out = run(InterpolateBads(), block(data, info(bads))).data
    raw = mne_raw(data, bads)
    good = [POS[lb] for lb in EEG if lb not in bads]
    centre, _ = fit_sphere(np.array(good))
    raw.interpolate_bads(reset_bads=False, origin=tuple(centre), verbose="error")
    np.testing.assert_allclose(out[: len(EEG)], raw.get_data(), rtol=1e-6, atol=1e-12)
    good_idx = [i for i, lb in enumerate(EEG) if lb not in bads]
    np.testing.assert_array_equal(out[good_idx], data[good_idx])
    np.testing.assert_array_equal(out[-1], data[-1])  # EOG untouched
    # the smooth field is recovered at the bad electrodes
    for lb in bads:
        i = EEG.index(lb)
        assert np.corrcoef(out[i], data[i])[0, 1] > 0.95
    nothing_bad = run(InterpolateBads(), block(data, info())).data
    np.testing.assert_array_equal(nothing_bad, data)


def test_csd_matches_mne_and_keeps_bad_channels() -> None:
    data = eeg_data()
    out = run(CSD(), block(data, info(("O2",))), lambda2=1e-5, stiffness=4.0,
              n_legendre_terms=50).data  # fmt: skip
    good = [lb for lb in EEG if lb != "O2"]
    raw = mne_raw(data).pick(good)  # MNE wants the bad channel gone
    centre, radius = fit_sphere(np.array([POS[lb] for lb in good]))
    csd = mne.preprocessing.compute_current_source_density(
        raw, sphere=(*centre, radius), verbose="error")  # fmt: skip
    idx = [EEG.index(lb) for lb in good]
    np.testing.assert_allclose(out[idx], csd.get_data(), rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(out[EEG.index("O2")], data[EEG.index("O2")])
    np.testing.assert_array_equal(out[-1], data[-1])


@pytest.mark.parametrize("proc", [InterpolateBads(), CSD()])
def test_spatial_nodes_pass_empty_blocks(proc: object) -> None:
    """A delaying node upstream may emit empty blocks (spec 0.2.5)."""
    params = {} if isinstance(proc, InterpolateBads) else {
        "lambda2": 1e-5, "stiffness": 4.0, "n_legendre_terms": 50}  # fmt: skip
    empty = Block(np.zeros((len(EEG) + 1, 0)), info(("Cz",)), 0, 0.0)
    assert run(proc, empty, **params).n_samples == 0


@pytest.mark.parametrize("proc", [InterpolateBads(), CSD()])
def test_spatial_nodes_are_the_same_block_by_block(proc: object) -> None:
    b = block(eeg_data(800), info(("Cz",)))
    params = {} if isinstance(proc, InterpolateBads) else {
        "lambda2": 1e-5, "stiffness": 4.0, "n_legendre_terms": 50}  # fmt: skip
    check_blockwise_equivalence(proc, b, params, trials=3)  # type: ignore[arg-type]


# -- EOG regression ------------------------------------------------------------------


def test_eog_regression_removes_the_eye_and_keeps_the_brain() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    brain = rng.standard_normal((len(EEG), n)) * 1e-5
    eye = np.cumsum(rng.standard_normal(n)) * 1e-6  # slow, large
    eye -= eye.mean()
    weights = np.linspace(1.0, 0.05, len(EEG))  # frontal channels get more
    data = np.vstack([brain + weights[:, None] * eye, eye])
    b = block(data, info())
    stage = Stage(EOGRegression(), {"eog": "auto"})
    stage.fit(b, signal_fingerprint(b.info))
    assert stage.model is not None
    coef = np.asarray(stage.model.data["coef"], dtype=float).ravel()
    np.testing.assert_allclose(coef, weights, atol=0.02)
    out = stage.transform(b, signal_fingerprint(b.info)).block
    assert isinstance(out, Block)
    residual = out.data[: len(EEG)] - brain
    assert np.abs(residual).max() < 0.1 * np.abs(weights[0] * eye).max()
    np.testing.assert_array_equal(out.data[-1], data[-1])  # the EOG stays
    check_blockwise_equivalence(EOGRegression(), b, {"eog": "auto"}, calibration=b)


def test_eog_channels_are_found_by_type_name_or_list() -> None:
    by_type = info()
    assert split_channels(by_type, "auto")[1] == ["EOG"]
    named = SignalInfo(("Fz", "Cz", "VEOG", "heog"), SFREQ)  # no types given
    assert split_channels(named, "auto") == (["Fz", "Cz"], ["VEOG", "heog"])
    assert split_channels(named, "Fz") == (["Cz", "VEOG", "heog"], ["Fz"])
    with pytest.raises(ValueError, match="no EOG channel"):
        split_channels(SignalInfo(("Fz", "Cz"), SFREQ), "auto")
    with pytest.raises(ValueError, match="no channel named X1"):
        split_channels(named, "X1")


# -- registry and categories ---------------------------------------------------------


def test_every_built_in_processor_and_sink_has_a_category() -> None:
    registry = Registry.from_entry_points()
    for name in registry.names():
        manifest = registry.get(name).manifest
        if manifest.role in ("processor", "sink"):
            assert manifest.category, name
        assert manifest.references or name in (
            "meegxer.nodes.rereference", "meegxer.nodes.pick_channels",
            "meegxer.core.probe", "meegxer.core.feature_probe",
            "meegxer.core.review",  # a person marks; no method to cite
        ) or manifest.role in ("source", "sink"), name  # fmt: skip
    for name, category in (("meegxer.nodes.fir_filter", "filters"),
                           ("meegxer.nodes.csd", "reference"),
                           ("meegxer.nodes.eog_regression", "artifacts"),
                           ("meegxer.core.lsl_out", "streams")):  # fmt: skip
        assert registry.get(name).manifest.category == category


def test_a_pipeline_uses_the_nodes_own_channel_types_unless_told() -> None:
    from meegxer.core.graph import Graph
    from meegxer.core.stage import Stage as StageType

    registry = Registry.from_entry_points()
    nodes: list[dict[str, object]] = [
        {"id": "input", "node": "meegxer.core.synthetic", "params": {}},
        {"id": "eog", "node": "meegxer.nodes.eog_regression", "params": {}},
        {"id": "hp", "node": "meegxer.nodes.fir_filter", "params": {}},
    ]
    pipeline = {"spec_version": "0.2.9", "nodes": nodes}
    graph = Graph.from_dict(pipeline, registry)
    eog, hp = graph["eog"].impl, graph["hp"].impl
    assert isinstance(eog, StageType) and isinstance(hp, StageType)
    assert eog.channel_types == ("eeg", "eog") and hp.channel_types == ("eeg",)
    saved = {n["id"]: n for n in graph.to_dict()["nodes"]}
    assert "channel_types" not in saved["eog"]  # the default is not written
    nodes[1]["channel_types"] = ["eeg"]
    narrowed = Graph.from_dict(pipeline, registry)
    assert narrowed["eog"].impl.channel_types == ("eeg",)  # type: ignore[union-attr]
    assert {n["id"]: n for n in narrowed.to_dict()["nodes"]}["eog"][
        "channel_types"
    ] == ["eeg"]
