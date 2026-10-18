import json
from pathlib import Path

import numpy as np
from scipy import signal
from typer.testing import CliRunner

from meegxer.cli import app
from meegxer.core.sources import FileSource

runner = CliRunner()


def write_json(path: Path, nodes: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"spec_version": "0.2.0", "nodes": nodes}))
    return path


def test_nodes_lists_builtins() -> None:
    result = runner.invoke(app, ["nodes"])
    assert result.exit_code == 0
    assert "meegxer.nodes.iir_filter" in result.output


def test_run_filters_a_set_file(tmp_path: Path) -> None:
    """M2 exit: `meegxer run pipeline.json --input sub-01.set` -> filtered .set."""
    raw = tmp_path / "sub-01.set"
    make = write_json(tmp_path / "make.json", [
        {"node": "meegxer.core.synthetic", "params": {"duration": 10, "sfreq": 250}},
        {"node": "meegxer.core.set_writer", "params": {"path": str(raw)}},
    ])  # fmt: skip
    assert runner.invoke(app, ["run", str(make)]).exit_code == 0

    pipeline = write_json(tmp_path / "pipeline.json", [
        {"id": "input", "node": "meegxer.core.file"},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "notch", "node": "meegxer.nodes.notch"},
        {"id": "car", "node": "meegxer.nodes.rereference"},
    ])  # fmt: skip
    out = tmp_path / "sub-01_clean.set"
    result = runner.invoke(
        app, ["run", str(pipeline), "--input", str(raw), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "computed: input, hp, notch, car" in result.output

    before = FileSource({"path": str(raw)}).read(10**6)
    after = FileSource({"path": str(out)}).read(10**6)
    assert before is not None and after is not None
    assert after.info.ch_labels == before.info.ch_labels
    freqs, p_before = signal.welch(before.data, fs=250.0, nperseg=1000)
    _, p_after = signal.welch(after.data, fs=250.0, nperseg=1000)
    line = np.abs(freqs - 50.0) < 0.5
    assert p_after[:, line].mean() < 0.01 * p_before[:, line].mean()
    np.testing.assert_allclose(after.data.sum(axis=0), 0.0, atol=1e-9)  # CAR

    cache = tmp_path / "cache"
    args = ["run", str(pipeline), "-i", str(raw), "-o", str(out), "--overwrite",
            "--cache-dir", str(cache)]  # fmt: skip
    assert runner.invoke(app, args).exit_code == 0
    again = runner.invoke(app, [*args, "--block-size", "64"])
    assert "from cache: hp, notch, car" in again.output


def test_run_reports_errors(tmp_path: Path) -> None:
    pipeline = write_json(tmp_path / "p.json", [{"node": "meegxer.core.file"}])
    result = runner.invoke(app, ["run", str(pipeline), "-i", str(tmp_path / "no.set")])
    assert result.exit_code == 1 and "error" in result.output
    bad = runner.invoke(app, ["run", str(pipeline), "-o", str(tmp_path / "x.mp3")])
    assert bad.exit_code == 1 and "unsupported" in bad.output


def test_an_existing_output_is_kept_and_the_run_goes_on(tmp_path: Path) -> None:
    """A writer whose file exists skips its save with a note; the other writers
    still write and the command succeeds (--overwrite replaces it)."""
    out, table = tmp_path / "out.set", tmp_path / "bp.tsv"
    pipeline = write_json(tmp_path / "pipeline.json", [
        {"id": "input", "node": "meegxer.core.synthetic",
         "params": {"duration": 4, "sfreq": 250}},
        {"id": "bp", "node": "meegxer.nodes.band_power"},
        {"id": "table", "node": "meegxer.core.feature_tsv",
         "params": {"path": str(table)}},
    ])  # fmt: skip
    out.write_bytes(b"kept")
    result = runner.invoke(app, ["run", str(pipeline), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "skipped:" in result.output and f"wrote {out}" not in result.output
    assert out.read_bytes() == b"kept" and table.is_file()
    result = runner.invoke(app, ["run", str(pipeline), "-o", str(out), "--overwrite"])
    assert result.exit_code == 0, result.output
    assert out.read_bytes() != b"kept" and "skipped:" in result.output  # the table
