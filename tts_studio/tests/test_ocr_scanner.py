import unittest

from core.ocr_scanner import OcrScanner


class TestOcrScannerQuality(unittest.TestCase):
    def setUp(self):
        self.s = OcrScanner()

    def test_clean_text_is_gut(self):
        r = self.s.scan("Ein normaler deutscher Text ohne Probleme.")
        self.assertEqual(r["qualitaet"], "gut")
        self.assertEqual(r["gesamt_fehler"], 0)

    def test_empty_text_is_gut(self):
        r = self.s.scan("")
        self.assertEqual(r["qualitaet"], "gut")
        self.assertEqual(r["gesamt_fehler"], 0)

    def test_quality_mittel_threshold(self):
        # 6 unpaired » → mittel
        text = " ".join(f"»Wort{i}" for i in range(6))
        r = self.s.scan(text)
        self.assertEqual(r["qualitaet"], "mittel")

    def test_quality_gut_upper_boundary(self):
        # 5 errors → still gut
        text = " ".join(f"»Wort{i}" for i in range(5))
        r = self.s.scan(text)
        self.assertEqual(r["qualitaet"], "gut")

    def test_quality_schlecht_threshold(self):
        # 21 unpaired » → schlecht
        text = " ".join(f"»Wort{i}" for i in range(21))
        r = self.s.scan(text)
        self.assertEqual(r["qualitaet"], "schlecht")

    def test_quality_mittel_upper_boundary(self):
        # 20 errors → still mittel
        text = " ".join(f"»Wort{i}" for i in range(20))
        r = self.s.scan(text)
        self.assertEqual(r["qualitaet"], "mittel")


class TestOcrScannerUnpairedQuotes(unittest.TestCase):
    def setUp(self):
        self.s = OcrScanner()

    # --- Guillemets ---

    def test_paired_guillemets_no_error(self):
        r = self.s.scan('Er sagte: »Hallo.«')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 0)

    def test_unpaired_opening_guillemet(self):
        r = self.s.scan('Er sagte: »Hallo.')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 1)

    def test_unpaired_closing_guillemet(self):
        r = self.s.scan('Hallo.«')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 1)

    def test_nested_guillemets_balanced(self):
        r = self.s.scan('»Er sagte »Ja« dazu.«')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 0)

    def test_two_unpaired_openers(self):
        r = self.s.scan('»Eins und »Zwei')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 2)

    def test_extra_closer_counts_as_error(self):
        r = self.s.scan('»Eins.« und nochmal.«')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 1)

    # --- Typografische Anführungszeichen ---

    def test_paired_typographic_no_error(self):
        r = self.s.scan('\u201eHallo.\u201c')  # „Hallo."
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 0)

    def test_unpaired_typographic_open(self):
        r = self.s.scan('\u201eHallo.')  # „ ohne schließendes "
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 1)

    def test_unpaired_typographic_close(self):
        r = self.s.scan('Hallo.\u201c')  # " ohne öffnendes „
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 1)

    def test_multiple_paired_typographic(self):
        r = self.s.scan('\u201eEins.\u201c und \u201eZwei.\u201c')
        self.assertEqual(r["fehlertypen"]["unpaarige_anfuehrungszeichen"], 0)


class TestOcrScannerUnusualChars(unittest.TestCase):
    def setUp(self):
        self.s = OcrScanner()

    def test_pipe_detected(self):
        r = self.s.scan("Text | mit Pipe")
        self.assertGreaterEqual(r["fehlertypen"]["ungewoehnliche_kombinationen"], 1)

    def test_broken_bar_detected(self):
        r = self.s.scan("Text \u00a6 Bruchstrich")
        self.assertGreaterEqual(r["fehlertypen"]["ungewoehnliche_kombinationen"], 1)

    def test_control_char_detected(self):
        r = self.s.scan("Text\x01mit NUL")
        self.assertGreaterEqual(r["fehlertypen"]["ungewoehnliche_kombinationen"], 1)

    def test_normal_german_chars_clean(self):
        r = self.s.scan("Äpfel, Öl und Übung – täglich.")
        self.assertEqual(r["fehlertypen"]["ungewoehnliche_kombinationen"], 0)

    def test_tab_and_newline_not_flagged(self):
        # \t and \n are excluded from the unusual pattern
        r = self.s.scan("Spalte1\tSpalte2\nZeile2")
        self.assertEqual(r["fehlertypen"]["ungewoehnliche_kombinationen"], 0)


class TestOcrScannerFundstellen(unittest.TestCase):
    def setUp(self):
        self.s = OcrScanner()

    def test_fundstelle_has_required_fields(self):
        r = self.s.scan('»Unpaired')
        self.assertEqual(len(r["fundstellen"]), 1)
        f = r["fundstellen"][0]
        for key in ("zeile", "position", "typ", "kontext", "raw_kontext"):
            self.assertIn(key, f)

    def test_correct_line_number_first_line(self):
        r = self.s.scan('»Fehler')
        self.assertEqual(r["fundstellen"][0]["zeile"], 1)

    def test_correct_line_number_second_line(self):
        r = self.s.scan("Erste Zeile\n»Fehler")
        self.assertEqual(r["fundstellen"][0]["zeile"], 2)

    def test_gesamt_fehler_matches_sum_of_types(self):
        r = self.s.scan('»Fehler | und Pipe')
        total = sum(r["fehlertypen"].values())
        self.assertEqual(r["gesamt_fehler"], total)
        self.assertEqual(r["gesamt_fehler"], len(r["fundstellen"]))

    def test_kontext_contains_surrounding_text(self):
        r = self.s.scan('Davor »Unpaired danach')
        kontext = r["fundstellen"][0]["kontext"]
        self.assertIn("Davor", kontext)
        self.assertIn("Unpaired", kontext)

    def test_raw_kontext_has_no_ellipsis_for_short_text(self):
        r = self.s.scan('»X')
        raw = r["fundstellen"][0]["raw_kontext"]
        self.assertNotIn("…", raw)

    def test_typ_values_are_valid(self):
        valid = {
            "unpaarige_anfuehrungszeichen",
            "verdaechtige_zeichen",
            "ungewoehnliche_kombinationen",
        }
        r = self.s.scan('»Fehler | Pipe')
        for f in r["fundstellen"]:
            self.assertIn(f["typ"], valid)


class TestOcrScannerPosToLineCol(unittest.TestCase):
    def test_first_char(self):
        line, col = OcrScanner._pos_to_line_col("abc", 0)
        self.assertEqual(line, 1)
        self.assertEqual(col, 1)

    def test_after_newline(self):
        text = "abc\nxyz"
        line, col = OcrScanner._pos_to_line_col(text, 4)  # 'x'
        self.assertEqual(line, 2)
        self.assertEqual(col, 1)

    def test_third_line(self):
        text = "a\nb\nc"
        line, col = OcrScanner._pos_to_line_col(text, 4)  # 'c'
        self.assertEqual(line, 3)
        self.assertEqual(col, 1)


if __name__ == "__main__":
    unittest.main()
