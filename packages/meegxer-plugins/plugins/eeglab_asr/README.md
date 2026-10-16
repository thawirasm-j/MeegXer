# EEGLAB ASR plugin

Artifact Subspace Reconstruction run by EEGLAB's own `asr_calibrate` and
`asr_process` (from the [clean_rawdata](https://github.com/sccn/clean_rawdata)
plugin) inside a sandboxed GNU Octave container (the built-in nodes have no ASR).

```bash
cd packages/meegxer-plugins
docker build -f plugins/eeglab_asr/Dockerfile -t meegxer/eeglab-asr:0.1.0 .
uv run meegxer-plugin check meegxer/eeglab-asr:0.1.0
uv run meegxer-plugin add meegxer/eeglab-asr:0.1.0
```

## Licence

MeegXer's files here (`eeglab_asr.m`, `mx_yulewalk.m`, the Dockerfile) are
GPL-3.0-or-later. The image downloads clean_rawdata at a pinned commit while it is
built. clean_rawdata's `asr_calibrate.m` and `asr_process.m` carry the UC Regents
licence: free for educational, research and non-profit use; commercial use needs
permission from UC San Diego's Technology Transfer Office (invent@ucsd.edu). Its
other files are GPL. Whoever builds and runs this image is bound by those terms;
MeegXer itself does not contain or redistribute clean_rawdata. The ASR method may
also be covered by patents; check before commercial use.

## Behaviour

- **Calibration** (`Calibrate` on the strip, or a batch run) follows EEGLAB's
  `clean_asr`: the clean windows of the calibration data are chosen first
  (`clean_windows`, max 7.5 % bad channels, z in [-3.5, 5.5], 1 s windows), then
  `asr_calibrate`. `calibration = all` skips the selection. Calibrate on a minute or
  more of mostly clean data. Put a high-pass (about 1 Hz) before the strip.
- **Processing** feeds `asr_process` fixed chunks of `step_size` samples (default
  half a window, as `clean_asr`), aligned to the start of the stream, so the output
  does not depend on how the stream is cut into blocks (acceptance test 1: max
  deviation 0).
- **Delay:** ASR looks ahead half a window (0.25 s by default). The output lags
  the input by that much and says so by sample index, so probes and other channels
  stay aligned; at the end of a recording the plugin extrapolates the last samples
  like `clean_asr`.
- **Any sampling rate:** Octave has no `yulewalk`, which `asr_calibrate` needs for
  rates it has no precomputed spectral filter for. `mx_yulewalk.m` implements the
  modified Yule-Walker method; it reproduces EEGLAB's precomputed filters
  (100-512 Hz) to about 1e-10.
- **Concordance with EEGLAB** (acceptance test 6): on `tests/data/m6_eeg.npz` the
  streamed output equals `clean_asr`'s to float32 precision
  (`tools/make_m6_references.py` makes the reference).
