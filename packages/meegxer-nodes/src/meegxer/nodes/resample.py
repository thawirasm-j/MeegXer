"""Resampling to another rate: a streamed polyphase FIR, live and offline alike.

The rates' exact ratio ``up / down`` (``out = in · up / down``) gives every output
sample ``n`` its time on the input's clock, ``t = n · down / up`` input samples
(rule 8: index / rate is the same time on both sides). It is the sum of the input
samples around ``t`` weighted by a linear-phase low-pass (Kaiser window, cut-off
at half the lower rate; as ``scipy.signal.resample_poly``), so the output is not
delayed. Live, an output sample waits until the input reaches past ``t`` by the
filter's half-length; the node holds that much back (a delaying node, spec 0.2.5)
and hands the rest over at the end of a recording, the input taken as zero beyond
its ends. Each output sample is computed the same way whatever the blocks, so
block sizes never change the result.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from functools import cache
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy import signal

from meegxer.core.block import CHANNEL_TYPES, Block, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes._manifest import stream_manifest

MAX_FACTOR = 1000  # largest up or down factor (the filter grows with it)
_HALF_LEN_PER_FACTOR = 10  # as resample_poly
_KAISER_BETA = 5.0


def ratio(sfreq_in: float, sfreq_out: float) -> tuple[int, int]:
    """``(up, down)`` with ``sfreq_out / sfreq_in == up / down``, in lowest terms."""
    exact = Fraction(str(sfreq_out)) / Fraction(str(sfreq_in))
    if exact.numerator > MAX_FACTOR or exact.denominator > MAX_FACTOR:
        raise ValueError(
            f"{sfreq_in:g} Hz to {sfreq_out:g} Hz needs factors above {MAX_FACTOR} "
            f"({exact.numerator}/{exact.denominator}); choose a rate with a "
            "simpler ratio"
        )
    return exact.numerator, exact.denominator


@cache
def design(up: int, down: int) -> npt.NDArray[np.float64]:
    """The anti-alias filter on the up-sampled grid (centred, gain ``up``)."""
    half = _HALF_LEN_PER_FACTOR * max(up, down)
    taps = signal.firwin(2 * half + 1, 1.0 / max(up, down),
                         window=("kaiser", _KAISER_BETA))  # fmt: skip
    return np.asarray(taps * up, dtype=np.float64)


def _position(state: State) -> tuple[int, int, int]:
    """The buffer's first input sample, the stream's first, the next output's."""
    keys = ("start", "first", "next")
    return tuple(int(state[k]) for k in keys)  # type: ignore[return-value,arg-type]


class Resample(Processor):
    """Change the sampling rate (exact rational ratio, linear phase, no delay)."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.resample",
        "0.1.0",
        [
            {"name": "sfreq", "type": "float", "default": 250.0, "min": 1.0,
             "max": 100000.0, "unit": "Hz", "primary": True,
             "description": "The new sampling rate"},
        ],
        description="Change the sampling rate (polyphase FIR, no delay).",
        channel_types=list(CHANNEL_TYPES),  # every channel goes to the new rate
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        up, down = ratio(block.info.sfreq, float(params["sfreq"]))
        if up == down:  # already at that rate
            return TransformResult(block, {})
        h = design(up, down)
        half = (h.size - 1) // 2
        if state is None:  # the stream's first sample: output starts at its time
            first_in = block.sample_index
            state = {"buffer": np.zeros((block.info.n_channels, 0)),
                     "start": first_in, "first": first_in,
                     "next": -(-first_in * up // down)}  # fmt: skip
        buffer = np.hstack([np.asarray(state["buffer"]), block.data])
        start, first, n0 = _position(state)
        stop = start + buffer.shape[1]
        # an output sample needs the input up to floor((n·down + half) / up)
        n1 = max(n0, (stop * up - half - 1) // down + 1)
        out = self._compute(buffer, start, first, n0, n1, h, up, down, stop)
        keep_from = max(start, -(-(n1 * down - half) // up))  # first still needed
        state = {"buffer": buffer[:, keep_from - start:], "start": keep_from,
                 "first": first, "next": n1}  # fmt: skip
        info = self._out_info(block.info, up, down)
        ts = block.timestamp + (n0 * down / up - block.sample_index) / block.info.sfreq
        return TransformResult(Block(out, info, n0, ts), state)

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        if state is None:
            return None
        up, down = ratio(info.sfreq, float(params["sfreq"]))
        if up == down or "buffer" not in state:
            return None
        h = design(up, down)
        buffer = np.asarray(state["buffer"])
        start, first, n0 = _position(state)
        stop = start + buffer.shape[1]
        n1 = -(-stop * up // down)  # every output sample before the input's end
        out = self._compute(buffer, start, first, n0, n1, h, up, down, stop)
        return TransformResult(Block(out, self._out_info(info, up, down), n0, 0.0),
                               {**state, "next": n1})  # fmt: skip

    @staticmethod
    def _compute(
        buffer: npt.NDArray[np.float64],
        start: int,
        first: int,
        n0: int,
        n1: int,
        h: npt.NDArray[np.float64],
        up: int,
        down: int,
        stop: int,
    ) -> npt.NDArray[np.float64]:
        """Output samples ``[n0, n1)`` from input samples ``[start, stop)`` held in
        ``buffer``; input before ``first`` or from ``stop`` on counts as zero."""
        n_ch = buffer.shape[0]
        if n1 <= n0:
            return np.zeros((n_ch, 0))
        half = (h.size - 1) // 2
        n = np.arange(n0, n1)
        k_lo = -(-(n * down - half) // up)  # first input sample with a weight
        width = 2 * half // up + 1
        ks = k_lo[:, None] + np.arange(width)  # (outputs, taps)
        taps = n[:, None] * down - ks * up + half  # into h
        valid = (taps >= 0) & (taps < h.size) & (ks >= first) & (ks < stop)
        weights = np.where(valid, h[np.clip(taps, 0, h.size - 1)], 0.0)
        idx = np.clip(ks - start, 0, max(buffer.shape[1] - 1, 0))
        if buffer.shape[1] == 0:
            return np.zeros((n_ch, n.size))
        return np.asarray(np.einsum("cnk,nk->cn", buffer[:, idx], weights),
                          dtype=np.float64)  # fmt: skip

    @staticmethod
    def _out_info(info: SignalInfo, up: int, down: int) -> SignalInfo:
        rate = info.sfreq * up / down
        out = dataclasses.replace(info, sfreq=rate)
        # the anti-alias filter is a low-pass at half the lower rate
        return out.filtered(("lowpass", 0.0, min(info.sfreq, rate) / 2))
