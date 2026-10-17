"""User settings and remembered state (meegxer.core.settings)."""

import json
from pathlib import Path

import jsonschema
import pytest

from meegxer.core import offline, settings
from meegxer.core.control import DEFAULT_CONTROL


def test_defaults_are_valid_and_match_the_daemon_endpoint() -> None:
    jsonschema.validate(settings.defaults(), settings.schema())
    assert settings.DEFAULTS["general"]["daemon"] == DEFAULT_CONTROL


def test_missing_keys_take_their_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"appearance": {"theme": "light"}}))
    s, problems = settings.read_settings(path)
    assert problems == []
    assert s["appearance"] == {"theme": "light", "scale": 100}
    assert s["visualization"]["sequential"] == "viridis"


def test_bad_parts_are_dropped_and_reported_never_fatal(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "appearance": {"theme": "light", "scale": 333},
        "visualization": {"sequential": "rainbow", "spectrum": {"band": [8, "x"]}},
        "storage": {"cache_gb": 2},
        "colour": "blue",
    }))  # fmt: skip
    s, problems = settings.read_settings(path)
    assert s["appearance"]["theme"] == "light" and s["appearance"]["scale"] == 100
    assert s["visualization"]["sequential"] == "viridis"
    assert s["visualization"]["spectrum"]["band"] == [8.0, 13.0]
    assert s["storage"]["cache_gb"] == 2
    text = "\n".join(problems)
    for bad in ("appearance.scale", "visualization.sequential",
                "visualization.spectrum.band", "unknown setting colour"):  # fmt: skip
        assert bad in text
    path.write_text("{not json")
    s, problems = settings.read_settings(path)
    assert s == settings.defaults() and "could not be read" in problems[0]


def test_saving_round_trips_and_refuses_invalid_settings(tmp_path: Path) -> None:
    s = settings.defaults()
    settings.put(s, "visualization.diverging", "blue-black-red")
    path = settings.save_settings(s, tmp_path / "a" / "settings.json")
    assert settings.read_settings(path) == (s, [])
    settings.put(s, "appearance.scale", 7)
    with pytest.raises(Exception, match="7"):
        settings.save_settings(s, path)


def test_environment_wins_over_the_file_over_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(settings.CONFIG_ENV, str(tmp_path))
    monkeypatch.delenv("MEEGXER_OFFLINE_DIR", raising=False)
    monkeypatch.delenv("MEEGXER_OFFLINE_CACHE_GB", raising=False)
    assert offline.offline_dir().name == "meegxer-offline"  # default: temp folder
    s = settings.defaults()
    settings.put(s, "storage.offline_dir", str(tmp_path / "runs"))
    settings.put(s, "storage.cache_gb", 1.5)
    settings.save_settings(s)
    assert offline.offline_dir() == tmp_path / "runs"
    assert offline._cache_cap_bytes() == int(1.5e9)
    monkeypatch.setenv("MEEGXER_OFFLINE_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("MEEGXER_OFFLINE_CACHE_GB", "3")
    assert offline.offline_dir() == tmp_path / "env"
    assert offline._cache_cap_bytes() == int(3e9)


def test_remembered_state_is_separate_and_forgiving(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    assert settings.load_state(path) == {}
    settings.save_state({"span": 20}, path)
    assert settings.load_state(path) == {"span": 20}
    path.write_text("[1, 2]")
    assert settings.load_state(path) == {}
    settings.clear_state(path)
    assert not path.exists()
    settings.clear_state(path)  # nothing to clear: fine
