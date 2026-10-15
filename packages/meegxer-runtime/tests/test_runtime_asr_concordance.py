"""Acceptance test 6 for ASR: the EEGLAB ASR plugin, streamed block by block and
flushed, against EEGLAB's own clean_asr on the shared M6 recording.

Runs in the sandbox when MEEGXER_TEST_IMAGES lists meegxer/eeglab-asr:*, or
unsandboxed (development) when MEEGXER_CLEAN_RAWDATA points at a clean_rawdata
checkout and GNU Octave is installed. The reference comes from
tools/make_m6_references.py.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo, concat_blocks
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.runtime.adapter import plugin_class, probe_manifest
from meegxer.runtime.worker import ContainerLauncher, Launcher, LocalLauncher

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "packages" / "meegxer-runtime"
IMAGES = [i for i in os.environ.get("MEEGXER_TEST_IMAGES", "").split(",") if i]
ASR_IMAGE = next((i for i in IMAGES if "eeglab-asr" in i), None)
CLEAN_RAWDATA = os.environ.get("MEEGXER_CLEAN_RAWDATA")


def launcher() -> Launcher:
    if ASR_IMAGE:
        return ContainerLauncher(ASR_IMAGE)
    assert CLEAN_RAWDATA is not None
    octave = (f"octave-cli --no-gui --quiet --norc --no-history "
              f"--path {RUNTIME / 'sdk' / 'octave'} "
              f"--path {RUNTIME / 'plugins' / 'eeglab_asr'} "
              f"--path {CLEAN_RAWDATA} --eval eeglab_asr")  # fmt: skip
    return LocalLauncher(
        ("sh", "-c", f"exec 3>&1 1>&2; MEEGXER_PROTOCOL_FD=3 exec {octave}"),
        allow=True,
    )


@pytest.mark.skipif(
    not ASR_IMAGE and not (CLEAN_RAWDATA and shutil.which("octave-cli")),
    reason="needs the meegxer/eeglab-asr image or MEEGXER_CLEAN_RAWDATA + Octave",
)
def test_eeglab_asr_matches_eeglab_clean_asr_in_the_sandbox() -> None:
    data = np.load(ROOT / "tests" / "data" / "m6_eeg.npz")
    ref = np.load(ROOT / "tests" / "data" / "m6_ref_asr.npz")["data"].astype(float)
    x = data["data"].astype(float)
    info = SignalInfo(tuple(str(s) for s in data["labels"]), float(data["sfreq"]))
    rec = Block(x, info, 0, 0.0)
    fp = signal_fingerprint(info)
    run = launcher()
    options = {"start_timeout": 300, "timeout": 300}
    cls = plugin_class(probe_manifest(run, **options), run, worker_options=options)
    stage = Stage(cls())
    try:
        stage.fit(rec, fp)  # calibrate on the recording itself, as clean_asr does
        parts = []
        for start in range(0, rec.n_samples, 500):  # stream in 500-sample blocks
            out = stage.transform(rec.slice(start, min(start + 500, rec.n_samples)), fp)
            assert isinstance(out.block, Block)
            parts.append(out.block)
        tail = stage.flush(fp)
        assert tail is not None and isinstance(tail.block, Block)
        parts.append(tail.block)
    finally:
        stage.processor.close()
    cleaned = concat_blocks(parts)
    assert (cleaned.sample_index, cleaned.n_samples) == (0, rec.n_samples)
    correction = ref - x
    assert np.abs(correction).max() > 50e-6  # EEGLAB did remove the artifacts
    assert np.abs(cleaned.data - ref).max() < 1e-7  # within 0.1 uV of EEGLAB
