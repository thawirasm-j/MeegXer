"""Session files (spec/session.md): readable models, reload checks, paths."""

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
from meegxer.core.block import Block
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION, Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.registry import Registry
from meegxer.core.serialization import State, hash_state
from meegxer.core.session import (
    SessionError,
    adopt_models,
    build_session,
    load_session,
    model_status,
    read_session,
    session_schema,
    sink_outputs,
    state_from_json,
    state_to_json,
    write_session,
)

REPO = Path(__file__).resolve().parents[3]


class Center(Processor):
    """Fitted node: removes the per-channel mean of the calibration data."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": SPEC_VERSION, "name": "test.center", "version": "0.1.0",
        "type": "stream", "requires_fit": True,
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip
    fits = 0

    def fit(self, calibration: Block, params: Params) -> State:
        Center.fits += 1
        return {"mean": calibration.data.mean(axis=1), "note": "test"}

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        mean = model["mean"]
        assert isinstance(mean, np.ndarray)
        out = Block(block.data - mean[:, None], block.info, block.sample_index,
                    block.timestamp)  # fmt: skip
        return TransformResult(out, {})


class FromImage(Center):
    """Stands in for an external plugin adapter (provenance from its image)."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": SPEC_VERSION, "name": "lab.center", "version": "1.0.0",
        "type": "stream", "requires_fit": True, "image": "ghcr.io/lab/center:1",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip
    image_id = "sha256:" + "a" * 64

    @classmethod
    def provenance(cls) -> dict[str, Any]:
        return {"image": "ghcr.io/lab/center:1", "image_id": cls.image_id}


@pytest.fixture
def registry() -> Registry:
    reg = Registry.from_entry_points()
    reg.add(Center)
    reg.add(FromImage)
    return reg


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((3, 1000)) * 1e-5 + 2e-5, info,
                          verbose="error")  # fmt: skip
    folder = tmp_path / "study"
    folder.mkdir()
    path = folder / "sub-01_raw.fif"
    raw.save(path, verbose="error")
    return path


def pipeline(path: Path, out: Path, node: str = "test.center") -> dict[str, Any]:
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "hp", "node": "meegxer.sterile.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "center", "node": node},
        {"id": "out", "node": "meegxer.core.set_writer",
         "params": {"path": str(out), "overwrite": True}},
    ]}  # fmt: skip


def run_and_save(registry: Registry, data: dict[str, Any], session: Path) -> Path:
    graph = Graph.from_dict(data, registry)
    BatchScheduler().run(graph)
    write_session(session, build_session(graph, outputs=sink_outputs(graph)))
    return session


def rerun(registry: Registry, session: Path, **options: Any) -> Any:
    loaded = load_session(read_session(session), registry, **options)
    fps = offline_input_fingerprints(loaded.graph)
    adopt_models(loaded, fps)
    status = model_status(loaded.graph, fps)
    before = Center.fits
    BatchScheduler().run(loaded.graph)
    return loaded, status, Center.fits - before


# -- readable models ---------------------------------------------------------------


def test_models_round_trip_bit_exact_as_plain_json() -> None:
    state: State = {
        "w": np.array([[0.1, -1e-300, math.nan], [math.inf, -math.inf, 1 / 3]]),
        "f32": np.array([1.1, 2.2], dtype=np.float32),
        "i": np.arange(6, dtype=np.int16).reshape(2, 3),
        "b": np.array([True, False]),
        "c": np.array([1 + 2j, math.nan - 1j]),
        "scalar0d": np.array(2.5),
        "empty": np.zeros((0, 3)),
        "nested": {"k": [1, 2.0, None, "x", math.nan, {"deep": np.ones(2)}]},
    }
    text = json.dumps(state_to_json(state), allow_nan=False)  # plain JSON
    back = state_from_json(json.loads(text))
    assert hash_state(back) == hash_state(state)
    assert back["f32"].dtype == np.float32 and back["i"].dtype == np.int16
    assert json.loads(text)["w"]["data"][0][2] == "NaN"


def test_package_schema_matches_spec() -> None:
    spec = json.loads((REPO / "spec" / "session.schema.json").read_text())
    assert session_schema() == spec


# -- save and reload ----------------------------------------------------------------


def test_reload_reuses_the_model_and_rewrites_the_same_file(
    registry: Registry, recording: Path, tmp_path: Path
) -> None:
    out = recording.parent / "clean.set"
    session = run_and_save(
        registry, pipeline(recording, out), recording.parent / "s.meegxer.json"
    )
    saved = json.loads(session.read_text())
    # paths inside the session folder are written relative to it
    assert saved["pipeline"]["nodes"][0]["params"]["path"] == "sub-01_raw.fif"
    assert saved["outputs"][0]["path"] == "clean.set"
    assert saved["models"]["center"]["data"]["mean"]["__array__"] == "float64"
    recorded = saved["outputs"][0]["sha256"]

    # the whole folder moves to another place (another machine)
    moved = tmp_path / "elsewhere" / "study"
    shutil.copytree(recording.parent, moved)
    (moved / "clean.set").unlink()
    loaded, status, fits = rerun(registry, moved / "s.meegxer.json")
    assert loaded.warnings == [] and status == {"center": "reused"} and fits == 0
    reread = read_session(moved / "s.meegxer.json")
    from meegxer.core.session import sha256_file

    assert sha256_file(Path(reread["outputs"][0]["path"])) == recorded


def test_version_mismatch_is_refused_unless_allowed(
    registry: Registry, recording: Path
) -> None:
    session = run_and_save(registry, pipeline(recording, recording.parent / "o.set"),
                           recording.parent / "s.meegxer.json")  # fmt: skip
    data = json.loads(session.read_text())
    data["nodes"]["center"]["version"] = "0.0.9"
    session.write_text(json.dumps(data))
    with pytest.raises(SessionError, match=re.escape("version 0.1.0, recorded 0.0.9")):
        load_session(read_session(session), registry)
    loaded, status, fits = rerun(registry, session, allow_version_mismatch=True)
    assert loaded.dropped_models == ["center"] and status == {} and fits == 1


def test_edited_model_is_refused_until_its_hash_is_removed(
    registry: Registry, recording: Path
) -> None:
    session = run_and_save(registry, pipeline(recording, recording.parent / "o.set"),
                           recording.parent / "s.meegxer.json")  # fmt: skip
    data = json.loads(session.read_text())
    data["models"]["center"]["data"]["mean"]["data"][0] = 0.0
    session.write_text(json.dumps(data))
    with pytest.raises(SessionError, match="changed after it was saved"):
        load_session(read_session(session), registry)
    del data["models"]["center"]["content_hash"]
    session.write_text(json.dumps(data))
    _, status, fits = rerun(registry, session)
    assert status == {"center": "reused"} and fits == 0


def test_hand_written_session_with_a_model_and_no_fingerprint(
    registry: Registry, recording: Path
) -> None:
    """A researcher writes the pipeline and a model by hand."""
    session = recording.parent / "hand.meegxer.json"
    session.write_text(json.dumps({
        "meegxer_session": "1.0",
        "pipeline": pipeline(Path("sub-01_raw.fif"), Path("hand.set")),
        "models": {"center": {
            "fitted_on": {"channels": ["Fz", "Cz", "Pz"], "sfreq": 250.0},
            "data": {"mean": {"__array__": "float64", "shape": [3],
                              "data": [0.0, 0.0, 0.0]}},
        }},
    }))  # fmt: skip
    loaded, status, fits = rerun(registry, session)
    assert status == {"center": "reused"} and fits == 0
    assert any("no fingerprint" in w for w in loaded.warnings)
    assert (recording.parent / "hand.set").is_file()  # relative to the session


def test_other_input_refits_and_changed_input_warns_or_refuses(
    registry: Registry, recording: Path
) -> None:
    session = run_and_save(registry, pipeline(recording, recording.parent / "o.set"),
                           recording.parent / "s.meegxer.json")  # fmt: skip
    raw = mne.io.read_raw_fif(recording, preload=True, verbose="error")
    raw._data[:] *= 2  # same file name, other content
    raw.save(recording, overwrite=True, verbose="error")
    with pytest.raises(SessionError, match="differs from the recorded one"):
        load_session(read_session(session), registry, strict=True)
    loaded, status, fits = rerun(registry, session)
    assert any("differs" in w for w in loaded.warnings)
    assert status == {"center": "needs refit"} and fits == 1  # rule 7


# -- external plugins ---------------------------------------------------------------


def test_plugin_provenance_is_recorded_and_checked(
    registry: Registry, recording: Path
) -> None:
    session = run_and_save(
        registry, pipeline(recording, recording.parent / "o.set", "lab.center"),
        recording.parent / "s.meegxer.json",
    )  # fmt: skip
    data = json.loads(session.read_text())
    assert data["nodes"]["center"]["provenance"] == FromImage.provenance()
    data["nodes"]["center"]["provenance"]["image_id"] = "sha256:" + "b" * 64
    session.write_text(json.dumps(data))
    with pytest.raises(SessionError, match="plugin image is sha256:a"):
        load_session(read_session(session), registry)


def test_missing_plugin_names_the_image_to_install(
    registry: Registry, recording: Path
) -> None:
    data = {
        "meegxer_session": "1.0",
        "pipeline": pipeline(recording, recording.parent / "o.set", "lab.absent"),
        "nodes": {"center": {"node": "lab.absent", "version": "1.0.0",
                             "provenance": {"image": "ghcr.io/lab/absent:1"}}},
    }  # fmt: skip
    with pytest.raises(
        SessionError, match=re.escape("meegxer-plugin add ghcr.io/lab/absent:1")
    ):
        load_session(data, registry)


def test_unsupported_format_and_invalid_sections_are_refused(
    registry: Registry, recording: Path
) -> None:
    base = {"pipeline": pipeline(recording, recording.parent / "o.set")}
    with pytest.raises(SessionError, match="not supported"):
        load_session({"meegxer_session": "2.0", **base}, registry)
    with pytest.raises(SessionError, match="session models"):
        load_session({"meegxer_session": "1.0", **base, "models": {"center": {}}},
                     registry)  # fmt: skip


def test_plain_pipeline_reads_as_a_session(registry: Registry, recording: Path) -> None:
    path = recording.parent / "p.json"
    path.write_text(json.dumps(pipeline(recording, recording.parent / "o.set")))
    loaded = load_session(read_session(path), registry)
    assert [n.id for n in loaded.graph.nodes] == ["input", "hp", "center", "out"]


def test_daemon_loads_a_session_and_gives_it_back(
    registry: Registry, recording: Path
) -> None:
    from meegxer.core.control import Daemon

    data = pipeline(recording, recording.parent / "o.set")
    data["nodes"] = data["nodes"][:3]  # live: no file writer needed
    graph = Graph.from_dict(data, registry)
    BatchScheduler().run(graph)
    session = build_session(graph, name="from offline")
    daemon = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*", registry=registry)
    try:
        reply = daemon.load_session(session)
        assert reply["models"] == ["center"] and reply["dropped_models"] == []
        back = daemon.get_session(name="again")
        assert back["name"] == "again"
        assert back["models"]["center"] == session["models"]["center"]
        assert back["nodes"] == session["nodes"]
    finally:
        daemon.close()
