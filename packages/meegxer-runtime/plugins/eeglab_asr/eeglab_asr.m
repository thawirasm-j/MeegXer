## Copyright (C) 2026 Thawirasm Jungrungrueang, TrimLAB, Prince of Songkla University
## SPDX-License-Identifier: GPL-3.0-or-later
##
## MeegXer plugin: Artifact Subspace Reconstruction with EEGLAB's own code.
##
## Runs asr_calibrate and asr_process from the clean_rawdata EEGLAB plugin, which
## the image fetches at build time (its ASR files are under the UC Regents licence:
## educational, research and non-profit use; see this plugin's README). Nothing of
## clean_rawdata is part of MeegXer.
##
## Calibration follows clean_asr: by default the clean windows of the calibration
## data are selected first (clean_windows), then asr_calibrate. Processing feeds
## asr_process fixed chunks of step_size samples aligned to the stream start, so
## the output does not depend on how the host cuts the stream into blocks. ASR
## looks ahead half a window: the output lags the input by that much (reported by
## sample index), and flush hands over the last samples by extrapolating the
## signal's end, as clean_asr does.
function eeglab_asr ()
  pkg load signal;
  manifest = struct (
    "spec_version", "0.2.5",
    "name", "eeglab.asr",
    "version", "0.1.0",
    "type", "stream",
    "requires_fit", true,
    "description", ["Artifact Subspace Reconstruction (EEGLAB clean_rawdata). ", ...
                    "Put a high-pass (e.g. 1 Hz) before it; calibrate on mostly ", ...
                    "clean data, ideally a minute or more."],
    "image", "meegxer/eeglab-asr:0.1.0",
    "inputs", {{struct("name", "in", "type", "signal")}},
    "outputs", {{struct("name", "out", "type", "signal")}},
    "params", {{ ...
      struct("name", "cutoff", "type", "float", "default", 20, "min", 1, ...
             "primary", true, "description", "Burst criterion: SD above calibration"), ...
      struct("name", "window_len", "type", "float", "default", 0.5, "min", 0.1, ...
             "unit", "s", "advanced", true, "description", "Statistics window"), ...
      struct("name", "max_dims", "type", "float", "default", 0.66, "min", 0, ...
             "max", 1, "advanced", true, ...
             "description", "Largest fraction of dimensions removed at once"), ...
      struct("name", "step_size", "type", "int", "default", 0, "min", 0, ...
             "advanced", true, ...
             "description", "Samples between statistics updates (0: half a window)"), ...
      struct("name", "calibration", "type", "enum", "default", "clean_windows", ...
             "choices", {{"clean_windows", "all"}}, "advanced", true, ...
             "description", "Calibrate on the clean windows (as EEGLAB) or on all")}});
  meegxer_plugin_run (manifest, @fit, @transform, @flush);
endfunction

function model = fit (data, info, params)
  srate = info.sfreq;
  X = data * 1e6;                                      # volts -> microvolts
  if (strcmp (params.calibration, "clean_windows"))
    sig = struct ("data", X, "srate", srate, "nbchan", rows (X), "pnts", columns (X),
                  "xmin", 0, "xmax", (columns (X) - 1) / srate, "trials", 1,
                  "chanlocs", struct ("labels", info.ch_labels), "etc", struct ());
    sig.event = struct ([]);                           # clean_windows without
    sig.urevent = struct ([]);                         # EEGLAB's pop_select
    try
      ref = clean_windows (sig, 0.075, [-3.5 5.5], 1);
      X = ref.data;
    catch err
      fprintf (stderr, "eeglab_asr: clean_windows failed (%s); calibrating on all\n",
               err.message);
    end_try_catch
  endif
  [B, A] = spectral_filter (srate);
  model = asr_calibrate (X, srate, params.cutoff, [], B, A, params.window_len,
                         [], [], [], 64);
endfunction

function [out, state, events, first] = transform (data, info, model, state, params)
  events = {};
  srate = info.sfreq;
  [L, step] = sizes (params, srate);
  X = data * 1e6;
  if (isempty (state))
    state = struct ("asr", model, "pending", zeros (rows (X), 0),
                    "tail", zeros (rows (X), 0), "fed", info.sample_index,
                    "start", info.sample_index);
  endif
  pending = [state.pending, X];
  n = floor (columns (pending) / step) * step;
  y = zeros (rows (X), n);
  asr = state.asr;
  for k = 0:step:n - 1
    [y(:, k+1:k+step), asr] = asr_process (pending(:, k+1:k+step), srate, asr,
                                           params.window_len, L / srate, step,
                                           params.max_dims, 64);
  endfor
  first = state.fed - L;                               # output lags by L samples
  tail = [state.tail, pending(:, 1:n)];
  state.tail = tail(:, max (1, end - L):end);          # last L+1 samples fed
  state.pending = pending(:, n+1:end);
  state.fed = state.fed + n;
  state.asr = asr;
  [out, first] = drop_warmup (y, first, state.start);
  out = out / 1e6;
endfunction

function [out, state, events, first] = flush (info, model, state, params)
  events = {};
  if (isempty (state))
    out = zeros (numel (info.ch_labels), 0);
    first = 0;
    return;
  endif
  srate = info.sfreq;
  [L, step] = sizes (params, srate);
  rem = state.pending;
  seen = [state.tail, rem];
  if (columns (seen) == 0)
    out = zeros (rows (rem), 0);
    first = state.fed;
    return;
  endif
  ## extrapolate the end by point reflection, as clean_asr does
  idx = (columns (seen) - 1):-1:(columns (seen) - L);
  idx = max (idx, 1);
  ext = 2 * seen(:, end) - seen(:, idx);
  [y, state.asr] = asr_process ([rem, ext], srate, state.asr, params.window_len,
                                L / srate, step, params.max_dims, 64);
  first = state.fed - L;
  [out, first] = drop_warmup (y, first, state.start);
  out = out / 1e6;
  state.pending = zeros (rows (rem), 0);
  state.fed = state.fed + columns (rem);
endfunction

function [L, step] = sizes (params, srate)
  L = round (params.window_len / 2 * srate);           # look-ahead, as clean_asr
  step = params.step_size;
  if (step <= 0)
    step = floor (srate * params.window_len / 2);
  endif
endfunction

function [out, first] = drop_warmup (y, first, start)
  ## The first L outputs of the stream belong before its first sample: keep what
  ## falls at or after START. An empty output sits where the next one begins.
  lo = max (first, start);
  out = y(:, min (columns (y), lo - first) + 1:end);
  first = lo;
endfunction

function [B, A] = spectral_filter (srate)
  ## asr_calibrate's default spectral weighting, designed for any sampling rate.
  [B, A] = mx_yulewalk (8, [[0 2 3 13 16 40 min(80, srate/2 - 1)] * 2 / srate, 1],
                        [3 0.75 0.33 0.33 1 1 3 3]);
endfunction
