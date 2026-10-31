"""Pipeline templates (M10.9): starting points for common studies, shipped as
pipeline files in ``meegxer/core/template_pipelines/``. The study templates
read a BIDS dataset and write into its BIDS derivatives; with ``sample`` they
read the sample dataset that fits them (``meegxer.core.samples``), its
subjects fetched when first run. The SSVEP BCI template is live: an LSL
amplifier, SSVEP detection and the keyboard strip (it has no sample; try it
with ``tools/ssvep_demo.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from meegxer.core import samples

FLANKER_STIMULI = ("stimulus/compatible/target_left, stimulus/compatible/target_right, "
                   "stimulus/incompatible/target_left, "
                   "stimulus/incompatible/target_right")  # fmt: skip


@dataclass(frozen=True)
class Template:
    name: str
    title: str
    description: str
    sample: str  # the sample dataset that fits it ("": none)
    # parameters set on its strips (by id) when it reads the sample
    sample_params: dict[str, dict[str, Any]] = field(default_factory=dict)


TEMPLATES: dict[str, Template] = {
    "erp": Template(
        "erp", "ERP study",
        "Band-pass 0.1-30 Hz, average reference, jumps marked, epochs -0.2 to "
        "0.8 s with a pre-stimulus baseline (epochs over a jump left out), and "
        "the epochs with their ERPs per label into the BIDS derivatives.",
        "erp_core", {"ep": {"events": FLANKER_STIMULI}},
    ),
    "rest": Template(
        "rest", "Resting-state spectra",
        "Band-pass 1-40 Hz, average reference, jumps cut out, then a power "
        "spectrum and band powers every 2 s, as feature tables in the BIDS "
        "derivatives.",
        "eegbci",
    ),
    "ica_cleaning": Template(
        "ica_cleaning", "ICA cleaning",
        "High-pass 1 Hz, average reference, ICA with ICLabel removing eye, "
        "muscle, heart, line-noise and channel-noise components, and the "
        "cleaned recording into the BIDS derivatives.",
        "eegbci",
    ),
    "ssvep_bci": Template(
        "ssvep_bci", "SSVEP BCI (keys)",
        "Live from an LSL amplifier: SSVEP detection by filter-bank CCA for "
        "targets flickering at 8, 10, 12 and 15 Hz, every 0.5 s over 2 s, and "
        "the keyboard strip pressing an arrow key for each detected target "
        "once armed by hand. Works best on the occipital channels (add Pick "
        "channels). MeegXer is not a medical device.",
        "",
    ),
}  # fmt: skip


def template_pipeline(name: str, *, sample: bool = False) -> dict[str, Any]:
    """The pipeline of template ``name``; with ``sample``, reading every
    subject of its sample dataset (the dataset's folder is made, nothing is
    downloaded until a run reads a subject)."""
    template = TEMPLATES[name]
    if sample and not template.sample:
        raise ValueError(f"the {template.title} template has no sample dataset")
    folder = resources.files("meegxer.core").joinpath("template_pipelines")
    text = folder.joinpath(f"{name}.json")
    pipeline: dict[str, Any] = json.loads(text.read_text(encoding="utf-8"))
    if sample:
        root = samples.create(template.sample)
        for node in pipeline["nodes"]:
            if node["id"] == "input":
                node["params"] = {**node.get("params", {}), "root": str(root)}
            if node["id"] in template.sample_params:
                node["params"] = {**node.get("params", {}),
                                  **template.sample_params[node["id"]]}  # fmt: skip
    return pipeline
