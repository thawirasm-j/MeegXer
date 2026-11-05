"""The Sterile switch (docs/STERILE.md): ``--sterile``/``--no-sterile`` over
``MEEGXER_STERILE`` over the ``general.sterile`` setting, off by default."""

from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest

from meegxer.core import settings
from meegxer.core.sterile import ENV, enabled, running


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MEEGXER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(ENV, raising=False)
    return tmp_path


def save(on: bool) -> None:
    data = settings.load_settings()
    data["general"]["sterile"] = on
    settings.save_settings(data)


def test_off_by_default_then_the_setting_the_variable_the_flag(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert enabled() is False
    save(True)
    assert enabled() is True
    monkeypatch.setenv(ENV, "off")
    assert enabled() is False  # the variable over the setting
    monkeypatch.setenv(ENV, "1")
    assert enabled() is True
    assert enabled(False) is False  # the flag over both
    monkeypatch.setenv(ENV, "maybe")
    with pytest.raises(ValueError, match="MEEGXER_STERILE='maybe'"):
        enabled()


def test_off_runs_locally_untouched(config: Path) -> None:
    """Off, a run's graph is not touched (no worker, no container)."""
    with running(object()) as isolation:
        assert isolation is None


def test_the_setting_is_schema_checked(config: Path) -> None:
    data = settings.load_settings()
    data["general"]["sterile"] = "yes"
    with pytest.raises(jsonschema.ValidationError, match="not of type 'boolean'"):
        settings.save_settings(data)
