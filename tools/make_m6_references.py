"""EEGLAB reference outputs for the M6 concordance checks (acceptance test 6).

Runs EEGLAB code in GNU Octave on ``tests/data/m6_eeg.npz`` and stores the result
next to it, so CI can compare MeegXer against EEGLAB without installing EEGLAB.

    uv run python tools/make_m6_references.py --clean-rawdata PATH/TO/clean_rawdata

ASR reference: clean_asr with its defaults (cutoff 20, 0.5 s windows, calibration
on the clean windows of the same data), the way EEGLAB's pop_clean_rawdata runs it.
Data go to EEGLAB in microvolts. clean_rawdata is not part of MeegXer (its ASR
files are under the UC Regents licence); pin the commit you used with --commit.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "m6_eeg.npz"

ASR_SCRIPT = r"""
addpath(CRD); addpath(fullfile(CRD, "private")); pkg load signal; warning("off", "all");
fid = fopen(IN, "r"); X = fread(fid, [NCH, Inf], "float32=>double"); fclose(fid);
labels = strsplit(LABELS, ",");
EEG = struct("data", X * 1e6, "srate", SRATE, "nbchan", NCH, "pnts", columns(X),
             "xmin", 0, "xmax", (columns(X) - 1) / SRATE, "trials", 1,
             "chanlocs", struct("labels", labels), "etc", struct());
EEG.event = struct([]); EEG.urevent = struct([]);  # clean_windows without EEGLAB
OUT_EEG = clean_asr(EEG, 20, 0.5, [], 0.66, 0.075, [-3.5 5.5], 1, false, false, 64);
fid = fopen(OUT, "w"); fwrite(fid, OUT_EEG.data / 1e6, "float64"); fclose(fid);
"""


def run_octave(script: str, **values: object) -> None:
    head = "".join(
        f'{k} = "{v}";\n' if isinstance(v, str) else f"{k} = {v};\n"
        for k, v in values.items()
    )
    subprocess.run(["octave-cli", "--no-gui", "--quiet", "--norc", "--eval",
                    head + script], check=True)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean-rawdata", type=Path, required=True)
    parser.add_argument("--commit", default="unknown", help="clean_rawdata commit")
    args = parser.parse_args()
    npz = np.load(DATA)
    x = npz["data"].astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        np.asfortranarray(x).T.tofile(inp)  # Octave reads column by column
        run_octave(ASR_SCRIPT, CRD=str(args.clean_rawdata.resolve()), IN=str(inp),
                   OUT=str(out), NCH=x.shape[0], SRATE=float(npz["sfreq"]),
                   LABELS=",".join(str(s) for s in npz["labels"]))  # fmt: skip
        ref = np.fromfile(out, dtype=np.float64).reshape(x.shape, order="F")
    dest = ROOT / "tests" / "data" / "m6_ref_asr.npz"
    np.savez_compressed(dest, data=ref.astype(np.float32),
                        clean_rawdata_commit=np.array(args.commit),
                        settings=np.array("clean_asr(EEG, 20, 0.5, [], 0.66, 0.075, "
                                          "[-3.5 5.5], 1); Octave"))  # fmt: skip
    changed = np.abs(ref - x).max(axis=1) * 1e6
    print(f"wrote {dest}; max change per channel (uV): {np.round(changed, 1)}")


if __name__ == "__main__":
    main()
