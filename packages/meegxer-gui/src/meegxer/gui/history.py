"""Undo and redo for the chain (M10.10): snapshots of the pipeline as it would
be saved (``PipelineDoc.to_pipeline(taps=False)``, as JSON).

The window records a snapshot after every edit (whatever made it: a strip
added, moved or removed, a setting, *Reads*, channel types, the recording's
settings). Several quick changes of one setting (a value scrolled or stepped)
are one step. Opening or making a pipeline starts a new history.
"""

from __future__ import annotations

import json
import time
from typing import Any

MERGE_SECONDS = 1.5  # changes of one setting closer than this are one step
LIMIT = 200  # steps kept


def _nodes(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        return []
    nodes = data.get("nodes", []) if isinstance(data, dict) else []
    return [n for n in nodes if isinstance(n, dict)]


def without_params(node: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k != "params"}


def same_structure(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """The same strips, in the same order and wiring: only settings differ."""
    return [without_params(n) for n in a] == [without_params(n) for n in b]


def changed_params(
    a: list[dict[str, Any]], b: list[dict[str, Any]]
) -> list[tuple[str, str, Any]]:
    """``(strip id, parameter, new value)`` of every setting ``b`` changes."""
    out = []
    for old, new in zip(a, b, strict=True):
        before, after = old.get("params", {}), new.get("params", {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                out.append((str(new["id"]), name, after.get(name)))
    return out


def edit_key(old: str, new: str) -> tuple[str, str] | None:
    """The one setting an edit changed, ``(strip id, parameter)``; None for
    any other edit."""
    a, b = _nodes(old), _nodes(new)
    if not a or not b or not same_structure(a, b):
        return None
    changed = changed_params(a, b)
    return (changed[0][0], changed[0][1]) if len(changed) == 1 else None


class EditHistory:
    """The chain's undo and redo stacks of snapshots; ``current`` is the
    chain as it is."""

    def __init__(self, limit: int = LIMIT) -> None:
        self.limit = limit
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.current = ""
        self._key: tuple[str, str] | None = None
        self._at = 0.0

    def reset(self, text: str) -> None:
        """A new history from ``text`` (a pipeline opened or made)."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current, self._key = text, None

    def record(self, text: str, now: float | None = None) -> bool:
        """After an edit: the chain is now ``text``. True when it is a step
        (the chain changed); a quick change of the setting the last step
        changed joins that step."""
        if text == self.current:
            return False
        now = time.monotonic() if now is None else now
        key = edit_key(self.current, text)
        joins = (key is not None and key == self._key and self.undo_stack
                 and now - self._at < MERGE_SECONDS)  # fmt: skip
        if not joins:
            self.undo_stack.append(self.current)
            del self.undo_stack[: -self.limit]
        self.redo_stack.clear()
        self.current, self._key, self._at = text, key, now
        return True

    def undo(self) -> str | None:
        """The chain before the last step (None: nothing to undo)."""
        if not self.undo_stack:
            return None
        self.redo_stack.append(self.current)
        self.current, self._key = self.undo_stack.pop(), None
        return self.current

    def redo(self) -> str | None:
        """The chain after the step last undone (None: nothing to redo)."""
        if not self.redo_stack:
            return None
        self.undo_stack.append(self.current)
        self.current, self._key = self.redo_stack.pop(), None
        return self.current


FIELDS = {
    "input": "what it reads",
    "channel_types": "channels processed",
    "strip_params": "settings on the strip",
    "recording": "recording settings",
    "pass_uncalibrated": "pass uncalibrated",
}


def describe_edit(old: str, new: str) -> str:
    """The edit that made ``new`` from ``old``, in a few words (the log's
    "Undo: …", "Redo: …")."""
    from meegxer.gui.labels import param_label  # labels import nothing from here
    from meegxer.gui.names import title

    a, b = _nodes(old), _nodes(new)

    def name(node: dict[str, Any]) -> str:
        return str(title(node["node"], node.get("params", {})))

    before, after = {n["id"]: n for n in a}, {n["id"]: n for n in b}
    added = [n for n in b if n["id"] not in before]
    removed = [n for n in a if n["id"] not in after]
    if a and b and a[0]["node"] != b[0]["node"]:
        return f"input is {name(b[0])}"
    if added and not removed:
        return "added " + ", ".join(name(n) for n in added)
    if removed and not added:
        return "removed " + ", ".join(name(n) for n in removed)
    if added or removed:
        return "chain changed"
    if [n["id"] for n in a] != [n["id"] for n in b]:
        moved = next(n for n, m in zip(b, a, strict=True) if n["id"] != m["id"])
        return f"moved {name(moved)}"
    if same_structure(a, b):
        changed = changed_params(a, b)
        strips = {sid for sid, _, _ in changed}
        if len(changed) == 1:
            sid, param, _ = changed[0]
            return f"{name(after[sid])}: {param_label(param)}"
        if len(strips) == 1:
            return f"{name(after[changed[0][0]])}: settings"
        return "settings"
    for old_node, new_node in zip(a, b, strict=True):
        if without_params(old_node) != without_params(new_node):
            keys = sorted(set(old_node) | set(new_node))
            fields = [
                FIELDS.get(k, k)
                for k in keys
                if k != "params" and old_node.get(k) != new_node.get(k)
            ]
            return f"{name(new_node)}: {', '.join(fields)}"
    return "chain changed"
