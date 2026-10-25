# Settings

MeegXer keeps your preferences in **`settings.json`** and what it remembers between
launches in **`state.json`**, both in the MeegXer config folder:

| System | Folder |
|---|---|
| Windows | `%APPDATA%\meegxer` |
| macOS, Linux | `$XDG_CONFIG_HOME/meegxer` (default `~/.config/meegxer`) |
| Any | `MEEGXER_CONFIG_DIR`, if set |

The installed plugins list (`plugins.json`) lives there too, and while the engine
runs, its access token (`daemon/*.token`, readable by you only), which the GUI and
CLI use to connect to it. While a .set or EDF file is being recorded, `recordings/`
holds a note of where (removed once the file is written), so the GUI can offer to
recover a recording a crash left unfinished.

In the GUI: **⚙** in the top bar, or **Ctrl+,**. Every change previews at once;
**OK** keeps it, **Cancel** goes back. **Reset to defaults** puts every setting back;
**Forget remembered view** clears `state.json`.

Settings are personal: nothing in them changes what a pipeline computes, so none of
it goes into pipelines, sessions or provenance.

## Where a value comes from

1. A command-line flag (e.g. `meegxer-gui --connect …`, `--theme …`).
2. An environment variable: `MEEGXER_OFFLINE_DIR`, `MEEGXER_OFFLINE_CACHE_GB`.
3. `settings.json`.
4. The default.

A file that is broken, or partly invalid, never stops MeegXer: the bad parts are
reported (in the log) and use their defaults. The format is checked against
`packages/meegxer-core/src/meegxer/core/schema/settings.schema.json`.

## What there is

| Page | Setting | Key | Default |
|---|---|---|---|
| General | Start in Live or Offline | `general.start_mode` | `live` |
| | Start in advanced mode | `general.advanced` | `false` |
| | Reopen the last pipeline (when the engine has none) | `general.reopen_last` | `true` |
| | Ask before closing while a pipeline runs | `general.confirm_quit_running` | `true` |
| | Engine address (used at the next start) | `general.daemon` | `tcp://127.0.0.1:5555` |
| Appearance | Theme: `dark`, `light`, `system`, or a theme file | `appearance.theme` | `dark` |
| | Text size (%) | `appearance.scale` | `100` |
| Visualization | Sequential map (band-power map) | `visualization.sequential` | `viridis` |
| | Diverging map (power-change map) | `visualization.diverging` | `blue-white-red` |
| | Frequency map (peak frequency) | `visualization.peak` | `plasma` |
| | Reverse sequential maps on a light theme | `visualization.flip_on_light` | `false` |
| | Time span (s) | `visualization.span` | `10` |
| | Live time axis: seconds before now, or time since start | `visualization.time_axis` | `before_now` |
| | Head plot band, power, remove 1/f, map, power range, live smoothing | `visualization.spectrum.*` | α (8–13 Hz), absolute, off, band power, 20 dB, 2 s |
| Review | Units: clinical (µV/mm and mm/s, on the screen's real size) or µV spacing and seconds per page | `review.units` | `uv_per_mm` |
| | Sensitivity (µV/mm), speed (mm/s) | `review.sensitivity`, `review.speed` | `7`, `30` |
| | Channel spacing (µV), page (s), for µV units | `review.spacing`, `review.page` | `100`, `10` |
| | Display montage: as recorded, average, longitudinal or transverse bipolar, Laplacian | `review.montage` | `average` |
| | Display high-pass, low-pass, notch (Hz; 0 is off) | `review.highpass`, `review.lowpass`, `review.notch` | `0`, `0`, `0` |
| Data & storage | Offline run files (empty: system temp folder) | `storage.offline_dir` | `""` |
| | Offline cache cap (GB) | `storage.cache_gb` | `5` |
| | Pipelines and sessions folder (file dialogs) | `storage.output_dir` | `""` |
| | Recordings folder (channel locations, montages, events) | `storage.data_dir` | `""` |
| Plugins & privacy | Installed plugins and scripts (remove, add a script, read the local-code notice) | (`plugins.json`) | |

**Colour maps** are the same in every theme by default, so figures stay comparable.
Sequential: viridis, cividis (colour-vision friendly), inferno, magma, plasma, turbo,
MeegXer. Diverging, blue (less) to red (more): through white or black (Kovesi's
CET-D1A and CET-D4), or through the theme's background (MeegXer). Diverging maps are
never flipped, since that would swap their meaning.

The **☀** button is a quick theme switch: it changes the theme setting.

## What is remembered

`state.json` holds the window's size and place, Friendly/Advanced, the time span,
the spectrum view (band, power, 1/f removal, colour, range, smoothing, head plot on or off), the
last folder and the last pipeline. A view's last state wins over its default
setting; changing that default in Settings replaces it, and **Forget remembered
view** returns everything to the settings.

Example `settings.json` (only what differs from the defaults is needed):

```json
{
  "appearance": {"theme": "system", "scale": 110},
  "visualization": {"sequential": "cividis"},
  "storage": {"offline_dir": "D:/meegxer-runs", "cache_gb": 20}
}
```
