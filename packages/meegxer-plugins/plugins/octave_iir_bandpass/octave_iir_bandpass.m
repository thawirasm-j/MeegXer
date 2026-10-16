## Reference Octave plugin: causal Butterworth band-pass with carried state.
##
## The filter state (one column of delay-line values per channel) lives in the
## host-owned STATE, so block-wise output equals one pass. The first block
## initialises the state to the steady-state response to its first sample, like
## the built-in meegxer.nodes.iir_filter.
function octave_iir_bandpass ()
  pkg load signal;
  manifest = struct (
    "spec_version", "0.2.1",
    "name", "examples.octave_iir_bandpass",
    "version", "0.1.0",
    "type", "stream",
    "description", "Causal Butterworth band-pass in GNU Octave (reference plugin).",
    "image", "meegxer/example-octave-iir-bandpass:0.1.0",
    "inputs", {{struct("name", "in", "type", "signal")}},
    "outputs", {{struct("name", "out", "type", "signal")}},
    "params", {{struct("name", "low", "type", "float", "default", 1.0, "min", 0, "unit", "Hz"), ...
                struct("name", "high", "type", "float", "default", 40.0, "min", 0, "unit", "Hz"), ...
                struct("name", "order", "type", "int", "default", 2, "min", 1, "max", 4, ...
                       "advanced", true)}});
  meegxer_plugin_run (manifest, [], @transform);
endfunction

function [out, state, events] = transform (data, info, model, state, params)
  nyq = info.sfreq / 2;
  if (! (0 < params.low && params.low < params.high && params.high < nyq))
    error ("need 0 < low < high < Nyquist (%g Hz)", nyq);
  endif
  [b, a] = butter (params.order, [params.low, params.high] / nyq);
  if (isempty (state))
    zi = steady_state_zi (b, a) * data(:, 1)';   # (n-1) x n_channels
  else
    zi = state.zi;
  endif
  ## dim 1 explicitly: a 1-sample block makes data' a row vector, which filter()
  ## would otherwise treat as one signal running across the channels.
  [y, zf] = filter (b, a, data', zi, 1);
  out = y';
  state = struct ("zi", zf);
  events = {};
endfunction

function zi = steady_state_zi (b, a)
  ## Same construction as scipy.signal.lfilter_zi.
  b = b / a(1); a = a / a(1);
  n = numel (a);
  IminusA = eye (n - 1) - compan (a)';
  B = b(2:end)' - a(2:end)' * b(1);
  zi = IminusA \ B;
endfunction
