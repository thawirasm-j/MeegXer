"""Number fields write Latin digits with a decimal point, whatever the
system's regional format (Windows check in a real window, error 6: on a Thai
Windows with native digits, Qt's spin boxes showed 2 and 8.0 Hz in Thai digits,
beside strip fields showing ``2``)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox


@pytest.fixture
def arabic_digits(qapp: QApplication) -> Iterator[None]:
    """A system format with other digits (Arabic, Egypt: Arabic-Indic digits),
    as a Thai Windows with native digits gives Qt."""
    QLocale.setDefault(QLocale(QLocale.Language.Arabic, QLocale.Country.Egypt))
    yield
    QLocale.setDefault(QLocale.c())  # as the fixture's app has it


def fields() -> tuple[str, str]:
    whole, real = QSpinBox(), QDoubleSpinBox()
    whole.setMaximum(100_000)
    whole.setValue(12_000)
    real.setValue(8.5)
    return whole.text(), real.text()


def test_number_fields_use_latin_digits(arabic_digits: None) -> None:
    from meegxer.gui.app import plain_numbers

    whole, real = fields()  # what the system format gives: not Latin digits
    assert not whole.isascii() and not real.isascii()
    plain_numbers()  # meegxer-gui does this before any widget exists
    assert fields() == ("12000", "8.50")
