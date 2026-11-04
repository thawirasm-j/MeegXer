"""The Sterile worker's entry point: ``python -m meegxer.plugins.sterile_worker``
(the Sterile image runs it; ``meegxer.plugins.sterile``)."""

from meegxer.plugins.sterile import serve

if __name__ == "__main__":
    raise SystemExit(serve())
