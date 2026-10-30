"""Sample data (M10.9): public EEG datasets as small BIDS datasets, fetched one
subject at a time, when a run first reads it.

Each sample is a BIDS dataset folder (``sample_dir()/<name>``) that lists every
subject in its ``meegxer_sample.json`` before any is downloaded. A BIDS input
over it sees all the listed recordings (``find_recordings``); reading one
(``BidsSource``) fetches that subject with MNE's dataset fetcher and writes it
into the dataset with mne-bids (EEGLAB ``.set``), so later runs reuse it.

- ``erp_core``: ERP CORE, the flanker task of one subject (MNE
  ``mne.datasets.erp_core``; Kappenman et al., 2021).
- ``eegbci``: the EEG Motor Movement/Imagery dataset's baseline runs, eyes open
  (run 1) and eyes closed (run 2), for 109 subjects (MNE
  ``mne.datasets.eegbci``; Schalk et al., 2004; Goldberger et al., 2000).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meegxer.core import settings

SAMPLE_ENV = "MEEGXER_SAMPLE_DIR"
MARKER = "meegxer_sample.json"

Fetch = Callable[[str, "Path | None"], list[tuple[str, Any]]]


@dataclass(frozen=True)
class Sample:
    name: str
    title: str
    description: str
    subjects: tuple[str, ...]
    tasks: tuple[str, ...]
    size: str  # per subject, roughly (empty: not known)
    references: tuple[str, ...]
    fetch: Fetch  # (subject, downloads folder) -> [(task, mne Raw)]


def _erp_core(subject: str, downloads: Path | None) -> list[tuple[str, Any]]:
    import mne

    folder = Path(mne.datasets.erp_core.data_path(path=downloads, update_path=False))
    found = sorted(folder.rglob("*Flankers*_eeg.fif"))
    if not found:
        raise FileNotFoundError(f"no ERP CORE flanker file in {folder}")
    raw = mne.io.read_raw_fif(found[0], preload=True, verbose="error")
    if raw.info["line_freq"] is None:
        raw.info["line_freq"] = 60.0  # recorded in the USA
    return [("flanker", raw)]


# the baseline runs (BIDS names resting-state tasks rest…)
EEGBCI_RUNS = {"resteyesopen": 1, "resteyesclosed": 2}


def _eegbci(subject: str, downloads: Path | None) -> list[tuple[str, Any]]:
    import mne
    from mne.datasets import eegbci

    files = eegbci.load_data(int(subject), list(EEGBCI_RUNS.values()),
                             path=downloads, update_path=False)  # fmt: skip
    out = []
    for task, path in zip(EEGBCI_RUNS, files, strict=True):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        eegbci.standardize(raw)  # "Fc5." -> "FC5"
        raw.set_montage("standard_1005", verbose="error")
        raw.info["line_freq"] = 60.0  # recorded in the USA
        out.append((task, raw))
    return out


SAMPLES: dict[str, Sample] = {
    "erp_core": Sample(
        "erp_core", "ERP CORE flanker",
        "The flanker task of ERP CORE (one subject, 30 EEG channels and EOG): "
        "compatible and incompatible flankers, and the responses.",
        ("001",), ("flanker",), "",
        ("Kappenman ES, Farrens JL, Zhang W, Stewart AX, Luck SJ (2021). ERP "
         "CORE: An open resource for human event-related potential research. "
         "NeuroImage 225:117465. https://doi.org/10.1016/j.neuroimage.2020.117465",),
        _erp_core,
    ),
    "eegbci": Sample(
        "eegbci", "EEGBCI resting state",
        "The baseline runs of the EEG Motor Movement/Imagery dataset: one "
        "minute eyes open and one eyes closed, 64 channels, 109 subjects.",
        tuple(f"{k:03d}" for k in range(1, 110)), tuple(EEGBCI_RUNS),
        "about 2.6 MB",
        ("Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR "
         "(2004). BCI2000: a general-purpose brain-computer interface (BCI) "
         "system. IEEE Trans Biomed Eng 51(6):1034-1043. "
         "https://doi.org/10.1109/TBME.2004.827072",
         "Goldberger AL, Amaral LAN, Glass L, et al. (2000). PhysioBank, "
         "PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220. "
         "https://doi.org/10.1161/01.CIR.101.23.e215"),
        _eegbci,
    ),
}  # fmt: skip


def sample_dir() -> Path:
    """Where the sample datasets go: ``MEEGXER_SAMPLE_DIR``, else the
    ``storage.sample_dir`` setting, else ``meegxer-samples`` in MNE's data
    folder (where MNE keeps its downloads)."""
    root = settings.resolve("storage.sample_dir", SAMPLE_ENV)
    if root:
        return Path(root)
    import mne

    return Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data") / (
        "meegxer-samples"
    )


def _downloads() -> Path | None:
    """MNE's own data folder (None) unless a sample folder is set: then the
    downloads go in it too."""
    root = settings.resolve("storage.sample_dir", SAMPLE_ENV)
    return Path(root) / "downloads" if root else None


def create(name: str) -> Path:
    """The sample's BIDS dataset folder, made if new: its description and the
    list of its subjects (nothing is downloaded yet)."""
    sample = SAMPLES[name]
    root = sample_dir() / name
    marker = root / MARKER
    if marker.is_file():
        return root
    import mne_bids

    root.mkdir(parents=True, exist_ok=True)
    mne_bids.make_dataset_description(
        path=root, name=sample.title, dataset_type="raw",
        references_and_links=list(sample.references), overwrite=True,
        verbose=False,
    )  # fmt: skip
    (root / ".bidsignore").write_text(f"{MARKER}\n", encoding="utf-8")
    listing = {"sample": name, "subjects": list(sample.subjects),
               "tasks": list(sample.tasks)}  # fmt: skip
    marker.write_text(json.dumps(listing, indent=2) + "\n", encoding="utf-8")
    return root


def sample_of(root: str | Path) -> Sample | None:
    """The sample a dataset folder holds, or None for any other dataset."""
    try:
        listing = json.loads((Path(root) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return SAMPLES.get(str(listing.get("sample", "")))


def listed_recordings(
    root: Path,
    *,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    tasks: list[str] | None = None,
    runs: list[str] | None = None,
) -> list[dict[str, str]] | None:
    """Every recording a sample dataset lists, fetched or not, as BidsSource
    parameters (None if ``root`` is not a sample)."""
    sample = sample_of(root)
    if sample is None:
        return None
    if sessions or runs:  # the samples have neither
        return []
    return [{"root": str(root), "subject": s, "session": "", "task": t,
             "run": "", "datatype": "eeg"}
            for s in sample.subjects if not subjects or s in subjects
            for t in sample.tasks if not tasks or t in tasks]  # fmt: skip


def _path(root: Path, subject: str, task: str) -> Path:
    return root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_eeg.set"


def fetched(root: Path, subject: str) -> bool:
    sample = sample_of(root)
    return sample is not None and all(
        _path(root, subject, t).is_file() for t in sample.tasks
    )


def needs_fetch(params: dict[str, Any]) -> bool:
    """Whether reading this BIDS input downloads a sample subject first."""
    root = str(params.get("root", ""))  # not Path(""): that is the working folder
    subject = str(params.get("subject", ""))
    return bool(root and subject and sample_of(root) is not None
                and not fetched(Path(root), subject))  # fmt: skip


_lock = threading.Lock()  # one fetch at a time: mne-bids updates shared files


def ensure_recording(params: dict[str, Any]) -> bool:
    """Fetch the subject a BIDS input names into its sample dataset, if the
    dataset is a sample and the subject is not there yet. True if it fetched."""
    folder = str(params.get("root", ""))  # not Path(""): the working folder
    subject = str(params.get("subject", ""))
    sample = sample_of(folder) if folder else None
    if sample is None or not subject:
        return False
    root = Path(folder)
    if subject not in sample.subjects:
        raise ValueError(f"{sample.title} has no subject {subject!r}")
    with _lock:
        if fetched(root, subject):
            return False
        import mne_bids

        for task, raw in sample.fetch(subject, _downloads()):
            path = mne_bids.BIDSPath(subject=subject, task=task, root=root,
                                     datatype="eeg")  # fmt: skip
            mne_bids.write_raw_bids(raw, path, overwrite=True, allow_preload=True,
                                    format="EEGLAB", verbose=False)  # fmt: skip
    return True
