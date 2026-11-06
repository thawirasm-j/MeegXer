"""Script folders (docs/CUSTOM_SCRIPTS.md): a script made of several files,
its helpers imported and its data files read as usual, its entry file found,
hashed whole for sessions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph
from meegxer.core.probes import MemoryProbe
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage
from meegxer.plugins import index
from meegxer.plugins.adapter import plugin_class, probe_manifest
from meegxer.plugins.cli import main
from meegxer.plugins.conformance import format_report, run_conformance
from meegxer.plugins.script import (
    ScriptError,
    entry_file,
    folder_files,
    load_script,
    script_manifest,
)
from meegxer.plugins.worker import LocalLauncher

MAIN = '''"""Scale by the gain kept in a data file (a script folder)."""

from pathlib import Path

import numpy as np

from lib.scaling import scaled  # a helper module of the folder

GAIN = np.load(Path(__file__).parent / "data" / "gain.npy")  # a data file


def process(raw, offset: float = 0.0):
    """Scale, then shift.

    Parameters
    ----------
    offset : float
        Added after scaling (V).
    """
    return scaled(raw.get_data(), GAIN) + offset
'''


def lab_folder(root: Path, gain: float = 3.0) -> Path:
    folder = root / "lab_method"
    (folder / "lib").mkdir(parents=True)
    (folder / "data").mkdir()
    (folder / "main.py").write_text(MAIN)
    (folder / "lib" / "__init__.py").write_text("")
    (folder / "lib" / "scaling.py").write_text(
        "def scaled(x, gain):\n    return x * float(gain)\n"
    )
    np.save(folder / "data" / "gain.npy", np.array(gain))
    (folder / "README.md").write_text("How to use it.\n")
    return folder


def strip(folder: Path) -> type:
    command = index.script_command(folder)
    launcher = LocalLauncher(tuple(command), allow=True)
    origin = index.origin_of({"command": command, "script": str(folder)})
    return plugin_class(probe_manifest(launcher), launcher, origin=origin)


def test_a_script_folder_runs_with_its_helpers_and_data(tmp_path: Path) -> None:
    folder = lab_folder(tmp_path)
    assert entry_file(folder) == folder / "main.py"
    manifest = script_manifest(load_script(folder), folder)
    assert manifest["name"] == "script.lab_method"  # named after the folder
    assert [p["name"] for p in manifest["params"]] == ["offset"]
    cls = strip(folder)
    src = {"duration": 2.0, "n_channels": 3, "sfreq": 100.0}
    probe, raw = MemoryProbe(), MemoryProbe()
    graph = Graph.chain([SyntheticSource(src), Stage(cls(), {}), probe])
    try:
        BatchScheduler().run(graph)
    finally:
        graph.nodes[1].impl.processor.close()  # type: ignore[union-attr]
    BatchScheduler().run(Graph.chain([SyntheticSource(src), raw]))
    np.testing.assert_allclose(probe.signal().data, 3.0 * raw.signal().data,
                               rtol=1e-6, atol=1e-12)  # fmt: skip
    checks = run_conformance(LocalLauncher(tuple(index.script_command(folder)),
                                           allow=True))  # fmt: skip
    assert all(c.passed for c in checks), format_report(checks)


def test_the_entry_file_is_the_one_defining_an_entry_function(tmp_path: Path) -> None:
    folder = tmp_path / "two"
    folder.mkdir()
    (folder / "helpers.py").write_text("def scaled(x):\n    return x\n")
    with pytest.raises(ScriptError, match="found none"):
        entry_file(folder)
    (folder / "a.py").write_text("def process(raw):\n    return raw\n")
    assert entry_file(folder).name == "a.py"
    (folder / "b.py").write_text("def features(data):\n    return data\n")
    with pytest.raises(ScriptError, match=r"found a\.py and b\.py: keep one"):
        entry_file(folder)
    # read, not run: an entry file that would fail on import is still found
    (folder / "b.py").write_text("raise SystemExit\n")
    assert entry_file(folder).name == "a.py"


def test_hidden_files_and_caches_are_not_part_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = lab_folder(tmp_path)
    (folder / ".git").mkdir()
    (folder / ".git" / "HEAD").write_text("ref\n")
    (folder / "lib" / "__pycache__").mkdir()
    (folder / "lib" / "__pycache__" / "scaling.cpython-314.pyc").write_bytes(b"x")
    (folder / ".env").write_text("SECRET=1\n")
    names = [name for name, _ in folder_files(folder)]
    assert names == ["README.md", "data/gain.npy", "lib/__init__.py",
                     "lib/scaling.py", "main.py"]  # fmt: skip
    import meegxer.plugins.script as script

    monkeypatch.setattr(script, "MAX_FOLDER_BYTES", 100)
    with pytest.raises(ScriptError, match="move recordings and outputs out"):
        folder_files(folder)


def test_the_hash_covers_every_file_of_the_folder(tmp_path: Path) -> None:
    """Changing a helper or a data file changes the hash sessions record;
    a hidden file does not. A single file's hash is the one recorded before
    script folders existed."""
    folder = lab_folder(tmp_path)
    command = index.script_command(folder)
    first = index.code_sha256(command)
    (folder / ".notes").write_text("mine\n")
    assert index.code_sha256(command) == first
    np.save(folder / "data" / "gain.npy", np.array(4.0))
    second = index.code_sha256(command)
    assert second != first
    (folder / "lib" / "scaling.py").write_text("def scaled(x, g):\n    return x\n")
    assert index.code_sha256(command) not in (first, second)
    single = tmp_path / "one.py"
    single.write_text("def process(raw):\n    return raw\n")
    digest = hashlib.sha256(b"one.py\0" + hashlib.sha256(single.read_bytes()).digest())
    assert index.code_sha256(index.script_command(single)) == digest.hexdigest()
    # another local command's folder arguments are not script folders
    other = ["prog", "--data", str(folder)]
    assert index.code_entries(other) == []


def test_the_cli_adds_and_tries_a_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import mne

    folder = lab_folder(tmp_path)
    info = mne.create_info(["Fz", "Cz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.ones((2, 500)) * 1e-6, info, verbose="error")
    rec = tmp_path / "rec_raw.fif"
    raw.save(rec, verbose="error")
    assert main(["run", "--script", str(folder), "-i", str(rec)]) == 0
    assert "script.lab_method: 2 channels" in capsys.readouterr().out
    assert main(["add", "--script", str(folder), "--accept-risk"]) == 0
    (entry,) = [e for e in index.load_index()
                if e["manifest"]["name"] == "script.lab_method"]  # fmt: skip
    assert entry["script"] == str(folder.resolve())
    origin = index.origin_of(entry)
    assert origin["script"] == "lab_method"
    assert origin["sha256"] == index.code_sha256(entry["command"])
