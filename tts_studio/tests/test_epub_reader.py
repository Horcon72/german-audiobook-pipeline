"""Unit-Tests für core.epub_reader.EpubReader."""

import io
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.epub_reader import EpubReader, EpubReaderError, _TextExtractor


# ---------------------------------------------------------------------------
# Hilfsfunktionen zum Erstellen synthetischer EPUB-Dateien im Speicher
# ---------------------------------------------------------------------------

_CONTAINER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0"
    xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="{opf_path}"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

_OPF_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    {manifest_items}
  </manifest>
  <spine>
    {spine_items}
  </spine>
</package>"""


def _make_epub(
    chapters: list[tuple[str, str]],   # [(filename, html_content), ...]
    opf_dir: str = "OEBPS",
) -> bytes:
    """Erzeugt ein minimales gültiges EPUB als Bytes-Objekt."""
    buf = io.BytesIO()
    opf_path = f"{opf_dir}/content.opf"

    manifest_items = "\n    ".join(
        f'<item id="ch{i}" href="{fname}" media-type="application/xhtml+xml"/>'
        for i, (fname, _) in enumerate(chapters)
    )
    spine_items = "\n    ".join(
        f'<itemref idref="ch{i}"/>' for i in range(len(chapters))
    )
    opf_xml = _OPF_TEMPLATE.format(
        manifest_items=manifest_items,
        spine_items=spine_items,
    )

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "META-INF/container.xml",
            _CONTAINER_XML.format(opf_path=opf_path),
        )
        zf.writestr(opf_path, opf_xml)
        for fname, html in chapters:
            zf.writestr(f"{opf_dir}/{fname}", html)

    return buf.getvalue()


def _epub_file(chapters: list[tuple[str, str]], tmp_path: Path) -> str:
    data = _make_epub(chapters)
    p = tmp_path / "test.epub"
    p.write_bytes(data)
    return str(p)


# ---------------------------------------------------------------------------
# _TextExtractor
# ---------------------------------------------------------------------------

class TestTextExtractor(unittest.TestCase):
    def _extract(self, html: str) -> str:
        p = _TextExtractor()
        p.feed(html)
        return p.get_text()

    def test_plain_text_preserved(self):
        self.assertEqual(self._extract("<p>Hallo Welt</p>"), "Hallo Welt")

    def test_script_tag_ignored(self):
        html = "<p>Text</p><script>alert(1)</script><p>Ende</p>"
        result = self._extract(html)
        self.assertNotIn("alert", result)
        self.assertIn("Text", result)
        self.assertIn("Ende", result)

    def test_style_tag_ignored(self):
        html = "<style>body{color:red}</style><p>Inhalt</p>"
        result = self._extract(html)
        self.assertNotIn("body", result)
        self.assertIn("Inhalt", result)

    def test_br_inserts_newline(self):
        html = "Zeile 1<br/>Zeile 2"
        result = self._extract(html)
        self.assertIn("\n", result)

    def test_block_tags_add_newlines(self):
        html = "<p>Absatz 1</p><p>Absatz 2</p>"
        result = self._extract(html)
        self.assertIn("Absatz 1", result)
        self.assertIn("Absatz 2", result)
        self.assertIn("\n", result)

    def test_headings_extracted(self):
        html = "<h1>Kapitel 1</h1><p>Erster Absatz.</p>"
        result = self._extract(html)
        self.assertIn("Kapitel 1", result)
        self.assertIn("Erster Absatz.", result)

    def test_multiple_blank_lines_collapsed(self):
        html = "<p>A</p>\n\n\n\n<p>B</p>"
        result = self._extract(html)
        self.assertNotIn("\n\n\n", result)

    def test_nested_ignored_tag(self):
        # Nested script inside script: skip_depth must handle correctly
        html = "<p>Vor</p><script><script>inner</script></script><p>Nach</p>"
        result = self._extract(html)
        self.assertNotIn("inner", result)
        self.assertIn("Vor", result)
        self.assertIn("Nach", result)

    def test_empty_html_returns_empty(self):
        self.assertEqual(self._extract(""), "")

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(self._extract("   \n  "), "")


# ---------------------------------------------------------------------------
# EpubReader._find_opf
# ---------------------------------------------------------------------------

class TestFindOpf(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def _make_zf(self, container_xml: str) -> zipfile.ZipFile:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("META-INF/container.xml", container_xml)
        buf.seek(0)
        return zipfile.ZipFile(buf, "r")

    def test_finds_opf_path(self):
        xml = _CONTAINER_XML.format(opf_path="OEBPS/content.opf")
        with self._make_zf(xml) as zf:
            path = EpubReader()._find_opf(zf)
        self.assertEqual(path, "OEBPS/content.opf")

    def test_missing_container_raises(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            with self.assertRaises(EpubReaderError):
                EpubReader()._find_opf(zf)

    def test_no_rootfile_raises(self):
        xml = (
            '<?xml version="1.0"?>'
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles/>'
            '</container>'
        )
        with self._make_zf(xml) as zf:
            with self.assertRaises(EpubReaderError):
                EpubReader()._find_opf(zf)

    def test_rootfile_without_full_path_raises(self):
        xml = (
            '<?xml version="1.0"?>'
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles>'
            '<rootfile media-type="application/oebps-package+xml"/>'
            '</rootfiles>'
            '</container>'
        )
        with self._make_zf(xml) as zf:
            with self.assertRaises(EpubReaderError):
                EpubReader()._find_opf(zf)


# ---------------------------------------------------------------------------
# EpubReader.read  (Integrationstests mit synthetischen EPUBs)
# ---------------------------------------------------------------------------

class TestEpubReaderRead(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._tmp = Path(self._tmpdir)

    def _path(self, chapters: list[tuple[str, str]]) -> str:
        return _epub_file(chapters, self._tmp)

    def test_single_chapter_text_extracted(self):
        p = self._path([("ch1.xhtml", "<html><body><p>Erster Absatz.</p></body></html>")])
        result = EpubReader().read(p)
        self.assertIn("Erster Absatz.", result)

    def test_multiple_chapters_joined(self):
        p = self._path([
            ("ch1.xhtml", "<html><body><p>Kapitel 1</p></body></html>"),
            ("ch2.xhtml", "<html><body><p>Kapitel 2</p></body></html>"),
        ])
        result = EpubReader().read(p)
        self.assertIn("Kapitel 1", result)
        self.assertIn("Kapitel 2", result)

    def test_chapters_separated_by_blank_line(self):
        p = self._path([
            ("ch1.xhtml", "<html><body><p>A</p></body></html>"),
            ("ch2.xhtml", "<html><body><p>B</p></body></html>"),
        ])
        result = EpubReader().read(p)
        self.assertIn("\n\n", result)

    def test_empty_chapters_skipped(self):
        p = self._path([
            ("ch1.xhtml", "<html><body></body></html>"),
            ("ch2.xhtml", "<html><body><p>Inhalt</p></body></html>"),
        ])
        result = EpubReader().read(p)
        self.assertEqual(result.strip(), "Inhalt")

    def test_spine_order_preserved(self):
        p = self._path([
            ("ch1.xhtml", "<html><body><p>Erstes</p></body></html>"),
            ("ch2.xhtml", "<html><body><p>Zweites</p></body></html>"),
            ("ch3.xhtml", "<html><body><p>Drittes</p></body></html>"),
        ])
        result = EpubReader().read(p)
        pos1 = result.index("Erstes")
        pos2 = result.index("Zweites")
        pos3 = result.index("Drittes")
        self.assertLess(pos1, pos2)
        self.assertLess(pos2, pos3)

    def test_html_tags_stripped(self):
        p = self._path([
            ("ch1.xhtml", "<html><body><p><strong>Fett</strong> normal</p></body></html>")
        ])
        result = EpubReader().read(p)
        self.assertNotIn("<strong>", result)
        self.assertIn("Fett", result)
        self.assertIn("normal", result)

    def test_script_content_not_in_output(self):
        p = self._path([
            ("ch1.xhtml", (
                "<html><head><script>var x=1;</script></head>"
                "<body><p>Text</p></body></html>"
            ))
        ])
        result = EpubReader().read(p)
        self.assertNotIn("var x", result)
        self.assertIn("Text", result)

    def test_bad_zip_raises_epub_reader_error(self):
        path = str(self._tmp / "bad.epub")
        Path(path).write_bytes(b"this is not a zip file")
        with self.assertRaises(EpubReaderError):
            EpubReader().read(path)

    def test_opf_in_root_no_subdir(self):
        """EPUB mit OPF direkt im Root (kein Unterverzeichnis)."""
        buf = io.BytesIO()
        opf_xml = _OPF_TEMPLATE.format(
            manifest_items='<item id="ch0" href="ch0.xhtml" media-type="application/xhtml+xml"/>',
            spine_items='<itemref idref="ch0"/>',
        )
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "META-INF/container.xml",
                _CONTAINER_XML.format(opf_path="content.opf"),
            )
            zf.writestr("content.opf", opf_xml)
            zf.writestr("ch0.xhtml", "<html><body><p>Wurzelebene</p></body></html>")
        path = str(self._tmp / "root.epub")
        Path(path).write_bytes(buf.getvalue())
        result = EpubReader().read(path)
        self.assertIn("Wurzelebene", result)

    def test_unicode_content_preserved(self):
        p = self._path([
            ("ch1.xhtml", (
                "<html><body><p>"
                "\u00c4 \u00d6 \u00dc \u00df \u2013 "
                "\u201eAnf\u00fchrung\u201c"
                "</p></body></html>"
            ))
        ])
        result = EpubReader().read(p)
        self.assertIn("\u00c4", result)   # Ä
        self.assertIn("\u00df", result)   # ß
        self.assertIn("\u2013", result)   # –

    def test_missing_spine_item_skipped(self):
        """Spine referenziert Datei, die im Archiv fehlt → wird toleriert."""
        buf = io.BytesIO()
        opf_xml = _OPF_TEMPLATE.format(
            manifest_items=(
                '<item id="ch0" href="ch0.xhtml" media-type="application/xhtml+xml"/>\n'
                '    <item id="ch1" href="missing.xhtml" media-type="application/xhtml+xml"/>'
            ),
            spine_items='<itemref idref="ch0"/><itemref idref="ch1"/>',
        )
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("META-INF/container.xml", _CONTAINER_XML.format(opf_path="OEBPS/content.opf"))
            zf.writestr("OEBPS/content.opf", opf_xml)
            zf.writestr("OEBPS/ch0.xhtml", "<html><body><p>Vorhanden</p></body></html>")
        path = str(self._tmp / "partial.epub")
        Path(path).write_bytes(buf.getvalue())
        result = EpubReader().read(path)
        self.assertIn("Vorhanden", result)


if __name__ == "__main__":
    unittest.main()
