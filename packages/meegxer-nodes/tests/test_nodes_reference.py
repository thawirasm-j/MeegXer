import numpy as np
import pytest

from meegxer.core.block import Block, ChannelMismatchError, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.nodes.reference import Rereference

INFO = SignalInfo(("Fz", "Cz", "M1", "M2"), 100.0)
DATA = np.random.default_rng(0).standard_normal((4, 300))
BLOCK = Block(DATA, INFO, 0, 0.0)


def run(**params: object) -> Block:
    out = Stage(Rereference(), params).transform(BLOCK, signal_fingerprint(INFO))
    assert isinstance(out.block, Block)
    return out.block


def test_average_reference() -> None:
    out = run()
    np.testing.assert_allclose(out.data, DATA - DATA.mean(axis=0))
    np.testing.assert_allclose(out.data.sum(axis=0), 0.0, atol=1e-12)


def test_channel_reference_by_label() -> None:
    out = run(mode="channels", channels="M1, M2")
    np.testing.assert_allclose(out.data, DATA - DATA[2:4].mean(axis=0))
    single = run(mode="channels", channels="Cz")
    np.testing.assert_allclose(single.data[1], 0.0)


def test_missing_or_empty_reference_raises() -> None:
    with pytest.raises(ChannelMismatchError):
        run(mode="channels", channels="A1")
    with pytest.raises(ValueError):
        run(mode="channels", channels=" , ")


@pytest.mark.parametrize("params", [{}, {"mode": "channels", "channels": "M1,M2"}])
def test_reref_blockwise_equals_one_pass(params: dict[str, object]) -> None:
    """Acceptance test 1 for re-referencing."""
    check_blockwise_equivalence(Rereference(), BLOCK, params)


BAD_INFO = SignalInfo(("Fz", "Cz", "M1", "M2"), 100.0, bads=("Cz",))
BAD_BLOCK = Block(DATA, BAD_INFO, 0, 0.0)


def run_on(block: Block, **params: object) -> Block:
    out = Stage(Rereference(), params).transform(block, signal_fingerprint(block.info))
    assert isinstance(out.block, Block)
    return out.block


def test_average_excludes_bads_and_listed_channels() -> None:
    good = [0, 2, 3]
    np.testing.assert_allclose(run_on(BAD_BLOCK).data, DATA - DATA[good].mean(axis=0))
    with_bads = run_on(BAD_BLOCK, exclude_bads=False)
    np.testing.assert_allclose(with_bads.data, DATA - DATA.mean(axis=0))
    listed = run_on(BLOCK, exclude="M1,M2")
    np.testing.assert_allclose(listed.data, DATA - DATA[:2].mean(axis=0))
    both = run_on(BAD_BLOCK, exclude="M2")
    np.testing.assert_allclose(both.data, DATA - DATA[[0, 2]].mean(axis=0))
    assert both.info.bads == ("Cz",)  # bad channels are still re-referenced, kept


def test_exclusion_errors() -> None:
    with pytest.raises(ChannelMismatchError):
        run_on(BLOCK, exclude="Oz")
    with pytest.raises(ValueError, match="every channel"):
        run_on(BLOCK, exclude="Fz,Cz,M1,M2")
    with pytest.raises(ValueError, match="marked bad"):
        run_on(BAD_BLOCK, mode="channels", channels="Cz")
    ok = run_on(BAD_BLOCK, mode="channels", channels="Cz", exclude_bads=False)
    np.testing.assert_allclose(ok.data[1], 0.0)


def test_reref_with_bads_blockwise_equals_one_pass() -> None:
    check_blockwise_equivalence(Rereference(), BAD_BLOCK, {"exclude": "M2"})


TYPED = SignalInfo(("Fz", "Cz", "VEOG", "ECG", "Pz"), 100.0, bads=("Pz",),
                   ch_types=("eeg", "eeg", "eog", "ecg", "eeg"))  # fmt: skip
TYPED_BLOCK = Block(np.random.default_rng(1).standard_normal((5, 300)), TYPED, 0, 0.0)


def pick(**params: object) -> Block:
    from meegxer.nodes.reference import PickChannels

    stage = Stage(PickChannels(), params)
    out = stage.transform(TYPED_BLOCK, signal_fingerprint(TYPED))
    assert isinstance(out.block, Block)
    return out.block


def test_pick_keeps_eeg_by_default_and_types_on_request() -> None:
    assert pick().info.ch_labels == ("Fz", "Cz", "Pz")
    both = pick(keep_eog=True)
    assert both.info.ch_labels == ("Fz", "Cz", "VEOG", "Pz")  # stream order
    np.testing.assert_array_equal(both.data, TYPED_BLOCK.data[[0, 1, 2, 4]])
    assert both.info.ch_types == ("eeg", "eeg", "eog", "eeg")
    assert pick(keep_eeg=False, keep_ecg=True).info.ch_labels == ("ECG",)


def test_pick_by_name_and_drop_bads() -> None:
    assert pick(keep_labels="Pz, Fz").info.ch_labels == ("Fz", "Pz")
    out = pick(drop_bads=True)
    assert out.info.ch_labels == ("Fz", "Cz") and out.info.bads == ()
    assert pick().info.bads == ("Pz",)  # kept bads stay marked
    with pytest.raises(ChannelMismatchError):
        pick(keep_labels="Oz")  # a label the stream lacks
    with pytest.raises(ValueError, match="type switched off"):
        pick(keep_labels="VEOG")  # EOG is off
    with pytest.raises(ValueError, match="no channel is kept"):
        pick(keep_labels="Pz", drop_bads=True)


def test_pick_is_blockwise_equivalent() -> None:
    from meegxer.nodes.reference import PickChannels

    check_blockwise_equivalence(PickChannels(), TYPED_BLOCK, {"keep_eog": True})
    params = {"keep_labels": "Cz,Pz", "drop_bads": True}
    check_blockwise_equivalence(PickChannels(), TYPED_BLOCK, params)
