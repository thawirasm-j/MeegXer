"""The ERP tap (M10.3): a running mean and SEM per label (Welford's update), so
they match MNE's average and SciPy's SEM of those epochs however many frames are
lost; it starts over on a reset, another epoch shape, or a change upstream."""

from __future__ import annotations

import mne
import numpy as np
import pytest
import scipy.stats

from meegxer.core.block import FeatureBlock
from meegxer.core.display import FeatureFrame, decode_frame, encode_features
from meegxer.core.probes import ErpProbe

CHANNELS = ("Fz", "Cz", "Pz")
TIMES = tuple(round(k / 100, 9) for k in range(-20, 51))


def epoch(label: str, data: np.ndarray, at: int, times: tuple[float, ...] = TIMES
          ) -> FeatureBlock:  # fmt: skip
    return FeatureBlock(
        data[np.newaxis], ("epoch", "channel", "time"), (at, at + data.shape[1]),
        0.0, {"epoch": (label,), "channel": CHANNELS, "time": times}, unit="V",
    )  # fmt: skip


def test_the_running_erp_and_sem_are_mnes_of_each_label() -> None:
    rng = np.random.default_rng(0)
    labels = ["go", "stop", "go", "go", "stop", "go"]
    data = rng.standard_normal((len(labels), len(CHANNELS), len(TIMES))) * 1e-6
    probe = ErpProbe()
    assert probe.average() is None  # nothing yet
    for k, (label, x) in enumerate(zip(labels, data, strict=True)):
        probe.tap(epoch(label, x, 100 * k))
    erp = probe.average()
    assert erp is not None
    assert erp.dims == ("stat", "condition", "channel", "time") and erp.unit == "V"
    assert erp.coords["stat"] == ("mean", "sem")
    assert erp.coords["condition"] == ("go", "stop")
    assert probe.counts == {"go": 4, "stop": 2}
    assert erp.window == (0, 500 + len(TIMES))  # first to last epoch
    info = mne.create_info(list(CHANNELS), 100.0, "eeg")
    events = np.array([[100 * k, 0, 1 + (lb == "stop")]
                       for k, lb in enumerate(labels)])  # fmt: skip
    ids = {"go": 1, "stop": 2}
    epochs = mne.EpochsArray(data, info, events, tmin=-0.2, event_id=ids,
                             verbose="error")  # fmt: skip
    for k, label in enumerate(("go", "stop")):
        np.testing.assert_allclose(erp.data[0, k], epochs[label].average().data,
                                   rtol=1e-12, atol=1e-20)  # fmt: skip
        # the sample SD (n - 1); MNE's standard_error divides by n
        chosen = data[[lb == label for lb in labels]]
        np.testing.assert_allclose(erp.data[1, k], scipy.stats.sem(chosen, axis=0),
                                   rtol=1e-10, atol=1e-20)  # fmt: skip


def test_it_starts_over_on_reset_another_shape_or_a_change_upstream() -> None:
    probe = ErpProbe()
    x = np.ones((len(CHANNELS), len(TIMES)))
    probe.follow("fp-a")
    probe.tap(epoch("go", x, 0))
    probe.changed = False
    probe.follow("fp-a")  # the same input: kept
    assert probe.counts == {"go": 1} and not probe.changed
    probe.follow("fp-b")  # a change upstream: a mean across it would mix them
    assert probe.counts == {} and probe.changed
    erp = probe.average()
    assert erp is not None and erp.data.shape == (2, 0, 3, len(TIMES))  # no epochs
    probe.tap(epoch("go", x, 0))
    probe.reset()
    assert probe.counts == {}
    probe.tap(epoch("go", x, 0))
    shorter = TIMES[:30]
    probe.tap(epoch("stop", x[:, :30], 200, shorter))  # tmax changed: over
    assert probe.counts == {"stop": 1} and probe.times == shorter
    probe.tap(FeatureBlock(np.ones(3), ("channel",), (0, 1), 0.0))  # not an epoch
    assert probe.counts == {"stop": 1}


def test_an_erp_frame_carries_each_conditions_count() -> None:
    """Display protocol 0.4: ``counts`` on a features frame."""
    values = np.zeros((2, 3, 4))
    frame = FeatureFrame("tap_ep__epochs", 7, (0, 400), 100.0,
                         ("condition", "channel", "time"),
                         {"condition": ("go", "stop"), "channel": CHANNELS,
                          "time": (0.0, 0.01, 0.02, 0.03)}, values, "V",
                         counts=(12, 3))  # fmt: skip
    back = decode_frame(encode_features(frame))
    assert isinstance(back, FeatureFrame)
    assert back.counts == (12, 3) and back.coords["condition"] == ("go", "stop")
    plain = decode_frame(encode_features(FeatureFrame(
        "p", 0, (0, 1), 100.0, ("channel",), {}, np.zeros(3))))  # fmt: skip
    assert isinstance(plain, FeatureFrame) and plain.counts == ()


@pytest.mark.parametrize("n", [1, 50])
def test_the_mean_stays_exact_over_many_epochs(n: int) -> None:
    probe = ErpProbe()
    for k in range(n):
        probe.tap(epoch("go", np.full((3, len(TIMES)), float(k)), 10 * k))
    erp = probe.average()
    assert erp is not None
    np.testing.assert_allclose(erp.data[0, 0], (n - 1) / 2)
    if n == 1:  # one epoch: no spread to speak of
        assert np.isnan(erp.data[1, 0]).all()
    else:  # 0..n-1: sample SD sqrt(n (n + 1) / 12)
        np.testing.assert_allclose(erp.data[1, 0], np.sqrt(n * (n + 1) / 12 / n))
