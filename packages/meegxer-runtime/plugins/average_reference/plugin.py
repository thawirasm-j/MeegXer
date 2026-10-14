"""Reference Python plugin: common average reference (stateless).

Built on the MeegXer Python SDK; numpy only. Bad channels (info["bads"]) are left
out of the average when exclude_bads is on, matching the built-in node.
"""

from typing import ClassVar

from meegxer_plugin import Plugin, run


class AverageReference(Plugin):
    manifest: ClassVar[dict] = {
        "spec_version": "0.2.1",
        "name": "examples.average_reference",
        "version": "0.1.0",
        "type": "stream",
        "description": "Common average reference (reference Python plugin).",
        "image": "meegxer/example-average-reference:0.1.0",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [{"name": "exclude_bads", "type": "bool", "default": True}],
    }

    def transform(self, data, info, model, state, params):
        labels = info["ch_labels"]
        bads = set(info.get("bads", [])) if params["exclude_bads"] else set()
        keep = [i for i, label in enumerate(labels) if label not in bads] or list(
            range(len(labels))
        )
        return data - data[keep].mean(axis=0, keepdims=True), {}, []


if __name__ == "__main__":
    run(AverageReference())
