from meegxer.gui.colors import channel_style, position, wheel_color
from meegxer.gui.theme import load_theme

DARK, LIGHT = load_theme("dark"), load_theme("light")


def luminance(rgb: tuple[int, int, int]) -> float:
    return sum(rgb) / (3 * 255)


def test_positions_cover_10_05_and_old_names() -> None:
    assert position("Cz") is not None and position("cz") == position("Cz")
    assert position("T3") == position("T7")
    assert position("FP1") == position("Fp1")
    assert position("E21") is None


def test_vertex_is_neutral_white_dark_and_near_black_light() -> None:
    cz_dark = channel_style("Cz", DARK).color
    cz_light = channel_style("Cz", LIGHT).color
    assert luminance(cz_dark) > 0.9 and max(cz_dark) - min(cz_dark) < 20
    assert luminance(cz_light) < 0.2 and max(cz_light) - min(cz_light) < 20


def test_hue_follows_the_angle() -> None:
    right = wheel_color(0.0, 1.0, DARK.wheel)  # orange
    left = wheel_color(180.0, 1.0, DARK.wheel)  # blue
    assert right[0] > right[2] and left[2] > left[0]
    t7, t8 = channel_style("T7", DARK).color, channel_style("T8", DARK).color
    assert t7[2] > t7[0] and t8[0] > t8[2]  # left blue, right orange


def test_bipolar_mixes_and_unknown_is_dashed() -> None:
    mix = channel_style("T7-T8", DARK)
    assert not mix.dashed
    assert channel_style("EOG1", DARK).dashed
    assert channel_style("E33", DARK).dashed
    assert channel_style("Fp1-Nope", DARK).dashed
