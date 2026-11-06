"""Porting code into MeegXer (docs/PORTING.md): the templates of `meegxer-plugin
new` and every "After" recipe of the porting guide work as written (the
conformance suite), and `meegxer-plugin run` tries a script on a recording."""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pytest

from meegxer.plugins import index
from meegxer.plugins.cli import main
from meegxer.plugins.conformance import format_report, run_conformance
from meegxer.plugins.scaffold import KINDS, new_script
from meegxer.plugins.trial import try_script
from meegxer.plugins.worker import LocalLauncher

GUIDE = Path(__file__).parents[3] / "docs" / "PORTING.md"


def conformant(path: Path) -> None:
    checks = run_conformance(
        LocalLauncher(tuple(index.script_command(path)), allow=True)
    )
    failed = [c for c in checks if not c.passed]
    assert not failed, format_report(failed)


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], 250.0, "eeg")
    data = np.random.default_rng(3).standard_normal((4, 250 * 20)) * 1e-5
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_annotations(mne.Annotations([2.0, 9.0], [0.0, 0.0], ["go", "go"]))
    path = tmp_path_factory.mktemp("rec") / "rec_raw.fif"
    raw.save(path, verbose="error")
    return path


@pytest.mark.parametrize("kind", KINDS)
def test_every_template_works_as_written(tmp_path: Path, kind: str) -> None:
    conformant(new_script(f"my_{kind}", kind, tmp_path))


def test_the_guides_recipes_work_as_written(tmp_path: Path) -> None:
    recipes = re.findall(r"After:.*?```python\n(.*?)```", GUIDE.read_text(), re.S)
    assert len(recipes) == 5  # process, transform, calibrated, features, epochs
    for i, code in enumerate(recipes):
        path = tmp_path / f"recipe_{i}.py"
        path.write_text(code)
        conformant(path)


def test_new_refuses_bad_names_and_existing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="letters, digits and underscores"):
        new_script("my-filter", "process", tmp_path)
    with pytest.raises(ValueError, match="kind must be one of"):
        new_script("ok", "nonsense", tmp_path)
    new_script("mine", "process", tmp_path)
    with pytest.raises(FileExistsError, match="choose another name"):
        new_script("mine", "process", tmp_path)


def test_run_says_what_came_out_and_writes_it(tmp_path: Path, recording: Path) -> None:
    signal = new_script("band", "process", tmp_path)
    done = try_script(signal, recording, out=tmp_path / "out.set",
                      params={"high": 30.0})  # fmt: skip
    assert done.kind == "signal" and done.channels == ["Fz", "Cz", "Pz", "Oz"]
    assert done.n_samples == 5000 and done.sfreq == 250.0
    assert mne.io.read_raw_eeglab(tmp_path / "out.set", verbose="error").n_times == 5000
    assert "4 channels, 5000 samples at 250 Hz (20.0 s)" in done.summary()
    features = new_script("measures", "features", tmp_path)
    done = try_script(features, recording, out=tmp_path / "f.tsv")
    assert done.kind == "features" and done.features == ["line_length", "variance"]
    assert done.windows == 37  # (20 - 2) / 0.5 + 1
    assert (tmp_path / "f.tsv").read_text().startswith("window_start")
    with pytest.raises(ValueError, match=r"write a signal to \.edf or \.set"):
        try_script(signal, recording, out=tmp_path / "out.txt")


def test_run_reports_the_events_a_script_adds(tmp_path: Path, recording: Path) -> None:
    path = tmp_path / "marks.py"
    path.write_text(
        "def process(raw):\n"
        "    raw.annotations.append([1.0, 4.0], [0.0, 0.0], ['peak', 'peak'])\n"
        "    return raw\n"
    )
    done = try_script(path, recording)
    assert done.events["peak"] == 2 and "events: " in done.summary()


def test_run_tries_an_epochs_script_after_an_epochs_strip(
    tmp_path: Path, recording: Path
) -> None:
    script = new_script("reject", "epochs", tmp_path)
    out = tmp_path / "kept-epo.fif"
    done = try_script(script, recording, out=out, epochs={"tmax": 0.5})
    assert (done.kind, done.epochs_in, done.kept["go"]) == ("epochs", 2, 2)
    assert "kept 2 of 2 epochs (go 2) x 4 channels" in done.summary()
    written = mne.read_epochs(out, verbose="error")
    assert len(written) == 2 and written.tmax == pytest.approx(0.5)
    none = try_script(script, recording, params={"threshold": 1.0})  # µV
    assert (none.epochs_in, sum(none.kept.values())) == (2, 0)
    assert "kept 0 of 2 epochs x 4 channels" in none.summary()
    with pytest.raises(ValueError, match=r"write epochs to \.fif or \.set"):
        try_script(script, recording, out=tmp_path / "kept.tsv")


def test_the_cli(
    tmp_path: Path, recording: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new", "lab_filter", "--kind", "transform",
                 "--folder", str(tmp_path)]) == 0  # fmt: skip
    said = capsys.readouterr().out
    assert "wrote" in said and "meegxer-plugin run --script" in said
    script = tmp_path / "lab_filter.py"
    assert main(["run", "--script", str(script), "-i", str(recording),
                 "--param", "gain=2.0"]) == 0  # fmt: skip
    assert "script.lab_filter: 4 channels" in capsys.readouterr().out
    assert main(["new", "lab_reject", "--kind", "epochs",
                 "--folder", str(tmp_path)]) == 0  # fmt: skip
    capsys.readouterr()
    assert main(["run", "--script", str(tmp_path / "lab_reject.py"),
                 "-i", str(recording), "--epochs", "tmin=-0.1"]) == 0  # fmt: skip
    assert "kept 2 of 2 epochs" in capsys.readouterr().out


def test_run_tries_the_file_as_it_is_now_after_add(
    tmp_path: Path, recording: Path
) -> None:
    """Trying a script again after adding it runs the file (not a clash
    with the added copy of the same name)."""
    script = new_script("added", "epochs", tmp_path)
    assert main(["add", "--script", str(script), "--accept-risk"]) == 0
    assert try_script(script, recording).kept["go"] == 2
