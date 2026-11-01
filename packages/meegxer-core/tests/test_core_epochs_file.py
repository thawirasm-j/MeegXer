"""An epochs file as the File input: MNE's ``-epo.fif`` (``mne.read_epochs``)
and an epoched EEGLAB ``.set`` (``mne.read_epochs_eeglab``). The input joins
the epochs end to end (the signal), puts each epoch's label at its time 0
(the Events rail) and starts the epochs rail with the epochs themselves."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import FeatureBlock
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.probes import ErpProbe, FeatureProbe
from meegxer.core.recording import describe
from meegxer.core.registry import Registry
from meegxer.core.sources import FileSource

LABELS = ["Fz", "Cz", "Pz", "Oz"]
SFREQ = 100.0


def make_epochs(seed: int = 0) -> mne.Epochs:
    rng = np.random.default_rng(seed)
    n, length = 6, 81  # -0.2 to 0.6 s
    data = rng.standard_normal((n, len(LABELS), length)) * 1e-6
    codes = np.array([1, 2, 1, 2, 1, 1])
    events = np.column_stack([np.arange(n) * 200 + 20, np.zeros(n, int), codes])
    info = mne.create_info(LABELS, SFREQ, "eeg")
    return mne.EpochsArray(data, info, events, tmin=-0.2,
                           event_id={"target": 1, "standard": 2},
                           verbose="error")  # fmt: skip


def write(path: Path) -> mne.Epochs:
    epochs = make_epochs()
    if path.suffix == ".set":
        epochs.export(path, fmt="eeglab", verbose="error")
    else:
        epochs.save(path, verbose="error")
    return epochs


@pytest.fixture(params=["rec-epo.fif", "rec.set"])
def epochs_file(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, mne.Epochs]:
    path = tmp_path / str(request.param)
    return path, write(path)


def test_the_epochs_are_joined_with_their_labels(
    epochs_file: tuple[Path, mne.Epochs],
) -> None:
    path, epochs = epochs_file
    source = FileSource({"path": str(path)})
    assert source.carries() == ["epochs"]
    data = epochs.get_data()
    assert source.n_samples == data.shape[0] * data.shape[2]
    block = source.read(10_000)
    assert block is not None
    np.testing.assert_allclose(
        block.data, np.concatenate(list(data), axis=1), rtol=1e-6, atol=1e-12
    )  # .set: single precision
    found = describe(source)
    starts = [round(e["onset"] * SFREQ) for e in found["events"]]
    assert starts == [k * 81 + 20 for k in range(6)]  # each epoch's time 0
    assert [e["label"] for e in found["events"]] == [
        "target", "standard", "target", "standard", "target", "target"]  # fmt: skip
    assert found["notes"][0].startswith(
        "Epochs file: 6 epochs of 0.81 s (target 4, standard 2), joined end to end"
    )


def run(path: Path, *taps: dict[str, object]) -> Graph:
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        *taps,
    ]}, Registry.from_entry_points())  # fmt: skip
    BatchScheduler(block_size=50).run(graph)  # epochs cross the blocks
    return graph


def test_the_input_starts_the_epochs_rail(
    epochs_file: tuple[Path, mne.Epochs],
) -> None:
    path, epochs = epochs_file
    graph = run(path,
                {"id": "tap", "node": "meegxer.core.feature_probe",
                 "input": "input:epochs", "params": {"keep_windows": 0}},
                {"id": "erp", "node": "meegxer.core.erp_probe",
                 "input": "input:epochs"})  # fmt: skip
    probe = graph["tap"].impl
    assert isinstance(probe, FeatureProbe)
    windows = [w for w in probe.windows if isinstance(w, FeatureBlock)]
    assert len(windows) == 6  # each once, however the blocks cut it
    data = epochs.get_data()
    for k, w in enumerate(windows):
        assert w.dims == ("epoch", "channel", "time") and w.window == (
            k * 81,
            (k + 1) * 81,
        )
        assert w.coords["channel"] == tuple(LABELS) and w.unit == "V"
        np.testing.assert_allclose(w.data[0], data[k], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(np.asarray(w.coords["time"], float), epochs.times)
    erp = graph["erp"].impl
    assert isinstance(erp, ErpProbe) and erp.counts == {"target": 4, "standard": 2}
    np.testing.assert_allclose(erp.means["target"],
                               epochs["target"].average().get_data(),
                               rtol=1e-5, atol=1e-12)  # fmt: skip


def test_a_continuous_recording_starts_no_epochs(tmp_path: Path) -> None:
    raw = mne.io.RawArray(
        np.zeros((2, 500)), mne.create_info(["Cz", "Pz"], SFREQ, "eeg"), verbose="error"
    )
    raw.export(tmp_path / "cont.set", fmt="eeglab", verbose="error")
    source = FileSource({"path": str(tmp_path / "cont.set")})
    assert source.carries() == []
    block = source.read(500)
    assert block is not None and source.rails(block) == {}


def test_an_import_review_crop_keeps_the_whole_epochs(tmp_path: Path) -> None:
    from meegxer.core.import_review import save_import_review

    path = tmp_path / "rec-epo.fif"
    epochs = write(path)
    source = FileSource({"path": str(path)})
    save_import_review(source, [], [], (1.0, 4.0), [], "me")
    tap: dict[str, object] = {"id": "tap", "node": "meegxer.core.feature_probe",
           "input": "input:epochs", "params": {"keep_windows": 0}}  # fmt: skip
    graph = run(path, tap)
    probe = graph["tap"].impl
    assert isinstance(probe, FeatureProbe)
    # samples 100-400 left: epochs 2 (162-243) and 3 (243-324) whole, from 62;
    # epoch 1 (81-162) and 4 (324-405) cut by the crop
    assert [w.window for w in probe.windows] == [(62, 143), (143, 224)]
    data = epochs.get_data()
    for k, w in enumerate(probe.windows, start=2):
        np.testing.assert_allclose(w.data[0], data[k])


def test_the_epochs_writer_writes_them_back(tmp_path: Path) -> None:
    path = tmp_path / "rec-epo.fif"
    epochs = write(path)
    out = tmp_path / "out-epo.fif"
    writer: dict[str, object] = {
        "id": "w", "node": "meegxer.core.epochs_writer", "input": "input:epochs",
        "params": {"path": str(out), "erp": False}}  # fmt: skip
    run(path, writer)
    back = mne.read_epochs(out, verbose="error")
    np.testing.assert_allclose(back.get_data(), epochs.get_data())
    assert [back.events[k, 2] for k in range(6)] == [
        back.event_id[x] for x in
        ("target", "standard", "target", "standard", "target", "target")]  # fmt: skip
