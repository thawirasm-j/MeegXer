"""The band-pass adapts to the mode: causal live, zero-phase offline (phase auto)."""

import numpy as np
import numpy.typing as npt
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph, GraphError
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage
from meegxer.nodes.filters import IIRFilter, IIRFilterZeroPhase

PARAMS = {"kind": "bandpass", "low": 4.0, "high": 12.0}


def source() -> SyntheticSource:
    return SyntheticSource({"duration": 8.0, "n_channels": 4, "sfreq": 250.0})


def batch_output(
    processor: IIRFilter | IIRFilterZeroPhase, **params: object
) -> npt.NDArray[np.float64]:
    probe = MemoryProbe()
    graph = Graph.chain([source(), Stage(processor, params), probe])
    BatchScheduler(block_size=64).run(graph)
    return probe.signal().data


@pytest.mark.parametrize("phase", ["auto", "zero"])
def test_offline_runs_zero_phase(phase: str) -> None:
    out = batch_output(IIRFilter(), **PARAMS, phase=phase)
    expected = batch_output(IIRFilterZeroPhase(), **PARAMS)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-18)


def test_offline_causal_stays_causal() -> None:
    out = batch_output(IIRFilter(), **PARAMS, phase="causal")
    zero = batch_output(IIRFilterZeroPhase(), **PARAMS)
    assert not np.allclose(out, zero)


def test_online_variant_is_causal_and_zero_is_refused() -> None:
    processor = IIRFilter()
    stage = Stage(processor, {**PARAMS, "phase": "auto"})
    assert stage.for_mode("online") is stage
    variant = stage.for_mode("offline")
    assert isinstance(variant.processor, IIRFilterZeroPhase)
    assert "phase" not in variant.params

    live = Graph.chain([source(), Stage(IIRFilter(), {"phase": "zero"})])
    with pytest.raises(GraphError, match="zero-phase"):
        RealtimeScheduler(live)


def test_switching_to_zero_phase_live_is_refused() -> None:
    graph = Graph.chain([source(), Stage(IIRFilter(), PARAMS)])
    sched = RealtimeScheduler(graph)
    sched.start()
    try:
        with pytest.raises(GraphError, match="zero-phase"):
            sched.set_params("iir_filter1", {"phase": "zero"})
        assert graph["iir_filter1"].impl.params["phase"] == "auto"  # unchanged
    finally:
        sched.stop()
