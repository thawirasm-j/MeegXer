# MeegXer display protocol (`meegxer.display/0.1`, unstable)

Decimated envelopes of every probe, published by the core on a ZeroMQ PUB socket
(default `tcp://127.0.0.1:5556`; `ping` on the control API returns the endpoint).
The stream is lossy by design (CLAUDE.md rule 2): the core never waits for a
viewer, and a slow subscriber simply misses frames.

## Message

One multipart message per probe per frame (about 30 frames per second):

1. **Topic:** probe id, UTF-8. Subscribe to a probe id to receive only that probe,
   or to the empty prefix for all.
2. **Header:** UTF-8 JSON:
   ```json
   {"protocol": "meegxer.display/0.1", "probe": "main", "seq": 41,
    "ch_labels": ["Fz", "Cz"], "sfreq": 500.0,
    "start_bin": 1200, "bin_size": 2, "n_bins": 8, "bypassed": []}
   ```
3. **Payload:** little-endian float32 array of shape `(2, n_channels, n_bins)`,
   C order. Row 0 is the minimum and row 1 the maximum of each bin, in volts.

`bypassed` lists nodes upstream of the probe that passed this data through
unprocessed because they were not calibrated yet (only nodes set to pass through;
by default such nodes block instead). A viewer should mark these frames.

## Alignment

Bin `k` covers absolute samples `[k * bin_size, (k + 1) * bin_size)`, so frames from
different probes of the same stream line up bin for bin (rule 8). The first bin of
a stream may cover fewer samples. `seq` increases by one per frame and probe; a gap
in `seq` means frames were dropped.
