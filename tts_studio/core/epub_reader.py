"""Liest EPUB-Dateien und extrahiert den Klartext."""

import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# HTML → Klartext
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimaler HTML-Parser: extrahiert sichtbaren Text, ignoriert Skripte/Styles."""

    _SKIP_TAGS = {"script", "style"}
    _BLOCK_TAGS = {
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "blockquote", "section", "article",
        "header", "footer", "nav", "aside",
    }

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br":
            self._parts.append("\n")
        elif tag in self._BLOCK_TAGS:
            # Sicherstellen, dass ein Zeilenumbruch vor dem Block steht
            if self._parts and not self._parts[-1].endswith("\n"):
                self._parts.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Mehrfache Leerzeilen → maximal eine Leerzeile
        raw = re.sub(r"\n[ \t]*\n[ \t\n]*", "\n\n", raw)
        return raw.strip()


# ---------------------------------------------------------------------------
# EPUB-Leser
# ---------------------------------------------------------------------------

class EpubReaderError(Exception):
    """Wird bei fehlerhaften oder nicht lesbaren EPUB-Dateien ausgelöst."""


class EpubReader:
    """Extrahiert den Klartext aus einer EPUB-Datei (EPUB 2 und 3)."""

    def read(self, path: str) -> str:
        """
        Liest ``path`` als EPUB-Datei und gibt den gesamten Klartext zurück.

        Die Kapitel werden in Spine-Reihenfolge zusammengesetzt und
        durch eine Leerzeile getrennt.

        Raises:
            EpubReaderError: Bei ungültigen EPUB-Dateien oder fehlenden Pflicht-Einträgen.
        """
        try:
            with zipfile.ZipFile(path, "r") as zf:
                opf_path = self._find_opf(zf)
                spine_paths = self._read_spine(zf, opf_path)
                chapters: list[str] = []
                for item_path in spine_paths:
                    try:
                        html_bytes = zf.read(item_path)
                    except KeyError:
                        # Toleranz bei fehlenden Spine-Einträgen
                        continue
                    text = self._extract_text(html_bytes)
                    if text.strip():
                        chapters.append(text)
                return "\n\n".join(chapters)
        except zipfile.BadZipFile as exc:
            raise EpubReaderError(f"Ungültige EPUB-Datei: {exc}") from exc
        except ET.ParseError as exc:
            raise EpubReaderError(
                f"Fehler beim Parsen des EPUB-Manifests: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _find_opf(self, zf: zipfile.ZipFile) -> str:
        """Liest META-INF/container.xml und gibt den Pfad zur OPF-Datei zurück."""
        try:
            container_xml = zf.read("META-INF/container.xml")
        except KeyError as exc:
            raise EpubReaderError(
                "META-INF/container.xml fehlt – keine gültige EPUB-Datei."
            ) from exc
        root = ET.fromstring(container_xml)
        # Namespace-tolerante Suche
        rootfile = root.find(
            ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
        )
        if rootfile is None:
            rootfile = root.find(".//rootfile")
        if rootfile is None:
            raise EpubReaderError(
                "META-INF/container.xml enthält keine rootfile-Referenz."
            )
        full_path = rootfile.attrib.get("full-path")
        if not full_path:
            raise EpubReaderError(
                "rootfile-Element hat kein full-path-Attribut."
            )
        return full_path

    def _read_spine(self, zf: zipfile.ZipFile, opf_path: str) -> list[str]:
        """Gibt die XHTML/HTML-Pfade in Spine-Reihenfolge zurück."""
        opf_dir = str(Path(opf_path).parent)
        if opf_dir == ".":
            opf_dir = ""

        opf_xml = zf.read(opf_path)
        root = ET.fromstring(opf_xml)

        # Namespace aus dem Root-Element bestimmen
        ns_uri = ""
        if root.tag.startswith("{"):
            ns_uri = root.tag[1 : root.tag.index("}")]
        prefix = f"{{{ns_uri}}}" if ns_uri else ""

        # Manifest: id → href (nur HTML/XHTML-Einträge)
        manifest: dict[str, str] = {}
        for item in root.iter(f"{prefix}item"):
            media_type = item.attrib.get("media-type", "")
            if "html" in media_type:
                item_id = item.attrib.get("id", "")
                href = item.attrib.get("href", "")
                if item_id and href:
                    manifest[item_id] = href

        # Spine: geordnete ID-Referenzen → vollständige Pfade
        spine: list[str] = []
        for itemref in root.iter(f"{prefix}itemref"):
            idref = itemref.attrib.get("idref", "")
            if idref in manifest:
                href = manifest[idref]
                full = f"{opf_dir}/{href}" if opf_dir else href
                spine.append(full)

        return spine

    @staticmethod
    def _extract_text(html_bytes: bytes) -> str:
        """Extrahiert sichtbaren Text aus einem HTML/XHTML-Bytes-Objekt."""
        html_str = html_bytes.decode("utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(html_str)
        return parser.get_text()
