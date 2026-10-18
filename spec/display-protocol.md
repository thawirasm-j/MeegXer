# MeegXer display protocol (`meegxer.display/0.3`, unstable)

Decimated envelopes of every probe, and the windows of every Features probe,
published by the core on a ZeroMQ PUB socket
(default `tcp://127.0.0.1:5556`; `ping` on the control API returns the endpoint).
The stream is lossy by design (CLAUDE.md rule 2): the core never waits for a
viewer, and a slow subscriber simply misses frames. A subscriber connects with the
daemon's token, as on the control socket (`control-api.md`, Authentication).

## Message

One multipart message per probe per frame (about 30 frames per second):

1. **Topic:** probe id, UTF-8. Subscribe to a probe id to receive only that probe,
   or to the empty prefix for all.
2. **Header:** UTF-8 JSON:
   ```json
   {"protocol": "meegxer.display/0.2", "probe": "main", "seq": 41,
    "ch_labels": ["Fz", "Cz"], "sfreq": 500.0,
    "start_bin": 1200, "bin_size": 2, "n_bins": 8, "bypassed": []}
   ```
3. **Payload:** little-endian float32 array of shape `(2, n_channels, n_bins)`,
   C order. Row 0 is the minimum and row 1 the maximum of each bin, in volts.

`bypassed` lists nodes upstream of the probe that passed this data through
unprocessed because they were not calibrated yet (only nodes set to pass through;
by default such nodes block instead). A viewer should mark these frames.

A probe on a rail of components (the ICs rail, e.g. `tap_ica1__ics` on
`ica1:ics`) adds each channel's class and its probability (ICLabel's most likely
class), in channel order:

```json
{"ch_classes": ["eye", "brain"], "ch_scores": [0.97, 0.81]}
```

## Feature frames (0.2)

A Features probe (`meegxer.core.feature_probe`, placed after a feature strip)
sends one message per feature window, with the same topic and three parts:

- **Header:**
  ```json
  {"protocol": "meegxer.display/0.2", "kind": "features", "probe": "tap_psd1",
   "seq": 7, "window": [1000, 1500], "sfreq": 500.0,
   "dims": ["channel", "frequency"],
   "coords": {"channel": ["Fz", "Cz"], "frequency": [1.0, 1.5]},
   "shape": [2, 2]}
  ```
- **Payload:** little-endian float32 array of `shape`, C order, one axis per
  entry of `dims`.

`window` is the range of source samples `[start, stop)` the window was computed
from and `sfreq` the source's rate, so `stop / sfreq` is the window's end in
seconds since the source's first sample (rule 8). Frames go out with the display
flush; when a flush holds more than 16 windows of one probe, only the newest 16
are sent (lossy, like envelopes).

0.3 adds the window's units, as a plugin's feature result states them
(`plugin-protocol.md`, spec 0.2.19): `unit`, or `unit_dim` and `units`, only when
the node states them:

```json
{"protocol": "meegxer.display/0.3", "kind": "features", "probe": "tap_psd1",
 "dims": ["channel", "frequency"], "unit": "V^2/Hz", "...": "..."}
```

Viewers should accept 0.1 messages (envelopes without classes) and 0.2 messages
(feature windows without units) as well.

## Alignment

Bin `k` covers absolute samples `[k * bin_size, (k + 1) * bin_size)`, so frames from
different probes of the same stream line up bin for bin (rule 8). The first bin of
a stream may cover fewer samples. `seq` increases by one per frame and probe; a gap
in `seq` means frames were dropped (envelopes and feature frames of one probe
share the count).
