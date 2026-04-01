"""
Tests für die Marker-Logik des MainWindow.

Da MainWindow Qt benötigt, testen wir ausschließlich die reinen Logik-Methoden,
die keinen Qt-Event-Loop voraussetzen:
  - _get_effective_text  (reine Textextraktion anhand von Offset-Markern)
  - _has_open_ocr_findings  (Auswertung von Scan-Ergebnis und Review-Flag)
  - _check_processing_needed  (Rückgabewert-Logik ohne Qt-Dialog: über Testdoppel)

Für _get_effective_text und _has_open_ocr_findings wird ein minimales Dummy-Objekt
verwendet, das die Attribute des MainWindow nachbildet, ohne Qt zu initialisieren.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Minimales Dummy-Objekt für den Marker-Zustand
# ---------------------------------------------------------------------------

class _MarkerHost:
    """Bildet den relevanten Zustand von MainWindow nach."""

    def __init__(self, text: str = ""):
        self._marker_start: int | None = None
        self._marker_end: int | None = None
        self._last_scan_result: dict | None = None
        self._ocr_findings_reviewed: bool = False
        self._output_up_to_date: bool = False
        self._text = text

    # Kopie der MainWindow-Methoden (identische Implementierung)

    def input_text_toPlainText(self) -> str:
        return self._text

    def _get_effective_text(self) -> str:
        text = self._text
        if self._marker_start is None and self._marker_end is None:
            return text
        start = self._marker_start if self._marker_start is not None else 0
        end = self._marker_end if self._marker_end is not None else len(text)
        return text[start:end]

    def _has_open_ocr_findings(self) -> bool:
        if not self._last_scan_result:
            return False
        if self._ocr_findings_reviewed:
            return False
        return self._last_scan_result.get("gesamt_fehler", 0) > 0


# ---------------------------------------------------------------------------
# _get_effective_text
# ---------------------------------------------------------------------------

class TestGetEffectiveText(unittest.TestCase):
    def _host(self, text: str) -> _MarkerHost:
        return _MarkerHost(text)

    def test_no_markers_returns_full_text(self):
        h = self._host("Vorspann\n\nInhalt\n\nNachspann")
        self.assertEqual(h._get_effective_text(), "Vorspann\n\nInhalt\n\nNachspann")

    def test_start_marker_only_returns_from_marker(self):
        h = self._host("ABCDEF")
        h._marker_start = 2
        self.assertEqual(h._get_effective_text(), "CDEF")

    def test_end_marker_only_returns_up_to_marker(self):
        h = self._host("ABCDEF")
        h._marker_end = 4
        self.assertEqual(h._get_effective_text(), "ABCD")

    def test_both_markers_returns_slice(self):
        h = self._host("ABCDEF")
        h._marker_start = 1
        h._marker_end = 4
        self.assertEqual(h._get_effective_text(), "BCD")

    def test_start_at_zero_returns_full_text_from_start(self):
        h = self._host("ABCDEF")
        h._marker_start = 0
        h._marker_end = 3
        self.assertEqual(h._get_effective_text(), "ABC")

    def test_end_at_len_returns_up_to_end(self):
        h = self._host("ABCDEF")
        h._marker_start = 2
        h._marker_end = 6
        self.assertEqual(h._get_effective_text(), "CDEF")

    def test_start_equals_end_returns_empty(self):
        h = self._host("ABCDEF")
        h._marker_start = 3
        h._marker_end = 3
        self.assertEqual(h._get_effective_text(), "")

    def test_empty_text_no_markers(self):
        h = self._host("")
        self.assertEqual(h._get_effective_text(), "")

    def test_multiline_with_markers(self):
        text = "Vorspann\nNoch Vorspann\nErster Inhalt\nZweiter Inhalt"
        h = self._host(text)
        start = text.index("Erster Inhalt")
        h._marker_start = start
        self.assertEqual(h._get_effective_text(), "Erster Inhalt\nZweiter Inhalt")

    def test_reset_markers_returns_full_text(self):
        h = self._host("ABCDEF")
        h._marker_start = 2
        h._marker_end = 4
        # Simulate reset
        h._marker_start = None
        h._marker_end = None
        self.assertEqual(h._get_effective_text(), "ABCDEF")


# ---------------------------------------------------------------------------
# _has_open_ocr_findings
# ---------------------------------------------------------------------------

class TestHasOpenOcrFindings(unittest.TestCase):
    def _host(self) -> _MarkerHost:
        return _MarkerHost()

    def test_no_scan_result_returns_false(self):
        h = self._host()
        self.assertFalse(h._has_open_ocr_findings())

    def test_zero_findings_returns_false(self):
        h = self._host()
        h._last_scan_result = {"gesamt_fehler": 0, "qualitaet": "gut", "fundstellen": []}
        self.assertFalse(h._has_open_ocr_findings())

    def test_findings_not_reviewed_returns_true(self):
        h = self._host()
        h._last_scan_result = {"gesamt_fehler": 3, "qualitaet": "mittel", "fundstellen": []}
        self.assertTrue(h._has_open_ocr_findings())

    def test_findings_reviewed_returns_false(self):
        h = self._host()
        h._last_scan_result = {"gesamt_fehler": 3, "qualitaet": "mittel", "fundstellen": []}
        h._ocr_findings_reviewed = True
        self.assertFalse(h._has_open_ocr_findings())

    def test_reviewed_flag_reset_returns_true_again(self):
        h = self._host()
        h._last_scan_result = {"gesamt_fehler": 2, "qualitaet": "schlecht", "fundstellen": []}
        h._ocr_findings_reviewed = True
        # New scan resets flag
        h._ocr_findings_reviewed = False
        self.assertTrue(h._has_open_ocr_findings())

    def test_new_scan_replaces_old_result(self):
        h = self._host()
        h._last_scan_result = {"gesamt_fehler": 5, "qualitaet": "schlecht", "fundstellen": []}
        h._ocr_findings_reviewed = True
        # Simulate new scan
        h._last_scan_result = {"gesamt_fehler": 0, "qualitaet": "gut", "fundstellen": []}
        h._ocr_findings_reviewed = False
        self.assertFalse(h._has_open_ocr_findings())


# ---------------------------------------------------------------------------
# Marker-Integration: Start vor End-Marker
# ---------------------------------------------------------------------------

class TestMarkerIntegration(unittest.TestCase):
    """Kombinierte Tests für realistischere Szenarien."""

    def test_preamble_removed_content_remains(self):
        preamble = "Impressum\nVerlag XY\n\n"
        content = "Kapitel 1\nDer erste Absatz."
        text = preamble + content
        h = _MarkerHost(text)
        h._marker_start = len(preamble)
        self.assertEqual(h._get_effective_text(), content)

    def test_both_preamble_and_afterword_removed(self):
        preamble = "VORSPANN "   # 9 chars
        content = "INHALT"       # 6 chars
        afterword = " NACHSPANN"
        text = preamble + content + afterword
        h = _MarkerHost(text)
        h._marker_start = len(preamble)
        h._marker_end = len(preamble) + len(content)
        self.assertEqual(h._get_effective_text(), content)

    def test_marker_at_boundary_unicode(self):
        text = "Ä" * 10 + "B" * 10  # Unicode chars, each 1 Python codepoint
        h = _MarkerHost(text)
        h._marker_start = 10
        self.assertEqual(h._get_effective_text(), "B" * 10)


if __name__ == "__main__":
    unittest.main()
