"""*New from template*: pick a pipeline template (``meegxer.core.templates``)
and, if wanted, its sample dataset (``meegxer.core.samples``)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from meegxer.core.samples import SAMPLES
from meegxer.core.templates import TEMPLATES


class TemplateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New from template")
        lay = QVBoxLayout(self)
        self.templates = QListWidget()
        for template in TEMPLATES.values():
            item = QListWidgetItem(template.title)
            item.setData(Qt.ItemDataRole.UserRole, template.name)
            self.templates.addItem(item)
        lay.addWidget(self.templates)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setObjectName("muted")
        lay.addWidget(self.description)
        self.sample = QCheckBox()
        lay.addWidget(self.sample)
        self.sample_note = QLabel()
        self.sample_note.setWordWrap(True)
        self.sample_note.setObjectName("muted")
        lay.addWidget(self.sample_note)
        note = QLabel("Without the sample, set the BIDS dataset on the Input "
                      "strip. The pipeline replaces the current one.")  # fmt: skip
        note.setWordWrap(True)
        lay.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.templates.currentRowChanged.connect(self._show)
        self.templates.setCurrentRow(0)
        self.resize(460, 360)

    def _show(self, row: int) -> None:
        template = list(TEMPLATES.values())[max(row, 0)]
        sample = SAMPLES[template.sample]
        self.description.setText(template.description)
        self.sample.setText(f"Use the sample data: {sample.title}")
        size = f" ({sample.size} each)" if sample.size else ""
        self.sample_note.setText(
            f"{sample.description} {len(sample.subjects)} subject(s); each is "
            f"downloaded the first time a run reads it{size}, the first one now."
        )

    def choice(self) -> tuple[str, bool]:
        """The template picked, and whether it reads its sample dataset."""
        item = self.templates.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)), self.sample.isChecked()
