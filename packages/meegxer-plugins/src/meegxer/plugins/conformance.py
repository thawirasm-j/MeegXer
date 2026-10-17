"""Plugin conformance suite (spec/plugin-protocol.md, docs/WORKPLAN.md M4).

Runs a plugin through the protocol and the processor contract and reports each
check. Used by ``meegxer-plugin check`` and by the tests of the reference plugins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from meegxer.core.block import Block, Event, FeatureBlock, concat_blocks
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.node import output_blocks
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.plugins.adapter import PluginProcessor, plugin_class
from meegxer.plugins.worker import Launcher, PluginWorker


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def labelled_trials(block: Block, params: Mapping[str, Any]) -> tuple[Event, ...]:
    """Trials for a plugin that fits on labelled trials (fit_events): every 2 s,
    alternating the labels of its ``classes`` parameter (default left, right)."""
    text = str(params.get("classes") or "left, right")
    labels = [c.strip() for c in text.split(",") if c.strip()] or ["left", "right"]
    step = max(1, round(2.0 * block.info.sfreq))
    starts = range(block.sample_index, block.stop_index - step, step)
    return tuple(Event(s, labels[k % len(labels)]) for k, s in enumerate(starts))


def test_signal(
    seconds: float = 8.0, n_channels: int = 8, sfreq: float = 250.0
) -> Block:
    block = SyntheticSource(
        {"duration": seconds, "n_channels": n_channels, "sfreq": sfreq, "seed": 7}
    ).read(10**7)
    assert block is not None
    return block


def run_conformance(
    launcher: Launcher,
    params: Mapping[str, object] | None = None,
    *,
    signal: Block | None = None,
    worker_options: Mapping[str, Any] | None = None,
) -> list[Check]:
    checks: list[Check] = []
    block = signal if signal is not None else test_signal()
    opts = dict(worker_options or {})

    def check(name: str, fn: Callable[[], str]) -> bool:
        try:
            checks.append(Check(name, True, fn() or ""))
            return True
        except Exception as exc:
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
            return False

    worker = PluginWorker(launcher, **opts)
    manifest: dict[str, Any] = {}
    cls: type[PluginProcessor] | None = None

    def handshake() -> str:
        manifest.update(worker.start())
        return (
            f"{manifest.get('name')} {manifest.get('version')} via {worker.sdk or '?'}"
        )

    def valid_manifest() -> str:
        nonlocal cls
        cls = plugin_class(manifest, launcher, worker_options=opts)
        return (
            f"availability={cls.manifest.availability}, fit={cls.manifest.requires_fit}"
        )

    def unknown_message() -> str:
        reply = worker.raw_exchange({"type": "definitely-not-a-message"})
        if reply.type != "error" or reply.header.get("error") != "bad_request":
            raise AssertionError(f"expected a bad_request error, got {reply.header}")
        if (
            worker.raw_exchange({"type": "hello", "spec_version": "0.2.1"}).type
            != "hello"
        ):
            raise AssertionError("plugin did not keep running after a bad request")
        return "answered bad_request and kept running"

    def shutdown() -> str:
        worker.close()
        if worker.last_exit_code not in (0, None):
            raise AssertionError(f"exit status {worker.last_exit_code}")
        return f"exit status {worker.last_exit_code}"

    if not check("handshake", handshake) or not check("manifest", valid_manifest):
        worker.close()
        return checks

    def credit() -> str:
        """Informative only: never fails."""
        authors = manifest.get("authors") or []
        refs = manifest.get("references") or []
        if authors or refs:
            return f"{len(authors)} author(s), {len(refs)} reference(s)"
        return "none given (optional): authors and references credit the work"

    check("credits", credit)
    check("unknown message", unknown_message)
    check("shutdown", shutdown)
    assert cls is not None
    plugin_cls = cls

    fp = signal_fingerprint(block.info)
    reference = Stage(plugin_cls(), params)
    one_pass: list[Any] = []
    trials = labelled_trials(block, reference.params) if (
        plugin_cls.manifest.fit_events) else ()  # fmt: skip

    def transform() -> str:
        if plugin_cls.manifest.requires_fit:
            reference.fit(block, fp, trials)
        result = reference.transform(block, fp)
        out = result.block
        if plugin_cls.manifest.kind == "analysis":
            windows = list(output_blocks(out))
            tail = reference.flush(fp)
            if tail is not None:
                windows += list(output_blocks(tail.block))
            if not windows or not all(isinstance(w, FeatureBlock) for w in windows):
                raise AssertionError("expected feature windows")
            for w in windows:
                assert isinstance(w, FeatureBlock)
                if not (block.sample_index <= w.window[0] <= w.window[1]
                        <= block.stop_index):  # fmt: skip
                    raise AssertionError(f"window {w.window} outside the input")
                if not np.all(np.isfinite(w.data)):
                    raise AssertionError("features contain NaN or inf")
            one_pass.extend(windows)
            return f"{len(windows)} feature windows, shape {windows[0].data.shape}"
        if not isinstance(out, Block):
            raise AssertionError("expected a Signal block")
        tail = reference.flush(fp)  # a lagging plugin hands over its last samples
        if tail is not None and isinstance(tail.block, Block):
            out = concat_blocks([out, tail.block])
        if (out.sample_index, out.n_samples) != (block.sample_index, block.n_samples):
            raise AssertionError("output must cover the same samples as the input")
        if not np.all(np.isfinite(out.data)):
            raise AssertionError("output contains NaN or inf")
        one_pass.append(out)
        return f"{out.info.n_channels} ch x {out.n_samples} samples"

    def blockwise() -> str:
        if plugin_cls.manifest.availability == "offline":
            return "skipped: offline-only plugin"
        processor = plugin_cls()
        try:
            worst = check_blockwise_equivalence(
                processor,
                block,
                params,
                calibration=block,
                events=trials,
                trials=3,
                rtol=1e-6,
                atol=1e-10,
            )
        finally:
            processor.close()
        return f"max deviation {worst:.3g}"

    def restart() -> str:
        stage = Stage(plugin_cls(), params)
        if reference.model is not None:
            stage.model = reference.model
        half = block.sample_index + block.n_samples // 2
        first = stage.transform(block.slice(block.sample_index, half), fp).block
        processor = stage.processor
        assert isinstance(processor, PluginProcessor)
        processor.worker._kill()  # simulate a crash between blocks
        second = stage.transform(block.slice(half, block.stop_index), fp).block
        if plugin_cls.manifest.kind == "analysis":
            got = [*output_blocks(first), *output_blocks(second)]
            tail = stage.flush(fp)
            if tail is not None:
                got += output_blocks(tail.block)
            processor.close()
            same = len(got) == len(one_pass) and all(
                isinstance(a, FeatureBlock) and isinstance(b, FeatureBlock)
                and a.window == b.window
                and np.allclose(a.data, b.data, rtol=1e-6, atol=1e-10)
                for a, b in zip(got, one_pass, strict=False)
            )  # fmt: skip
            if not same:
                raise AssertionError("feature windows after restart differ")
            return "worker killed mid-stream; relaunched and continued exactly"
        assert isinstance(first, Block) and isinstance(second, Block)
        parts = [first, second]
        tail = stage.flush(fp)
        if tail is not None and isinstance(tail.block, Block):
            parts.append(tail.block)
        joined = concat_blocks(parts)
        processor.close()
        if not np.allclose(joined.data, one_pass[0].data, rtol=1e-6, atol=1e-10):
            raise AssertionError("output after restart differs from one pass")
        return "worker killed mid-stream; relaunched and continued exactly"

    if check("transform", transform):
        check("acceptance test 1 (block-wise == one pass)", blockwise)
        if plugin_cls.manifest.availability != "offline":
            check("restart with host-held state", restart)
    reference.processor.close()
    return checks


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [
        f"{'PASS' if c.passed else 'FAIL'}  {c.name:<{width}}  {c.detail}"
        for c in checks
    ]
    verdict = "conformant" if all(c.passed for c in checks) else "NOT conformant"
    return "\n".join([*lines, verdict])
