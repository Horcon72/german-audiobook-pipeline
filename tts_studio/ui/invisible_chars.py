"""
Darstellung nicht druckbarer Zeichen in QPlainTextEdit-Feldern.

Zwei Mechanismen kombiniert:
1. QTextOption-Flags  → Qt rendert Leerzeichen (·), Tabs (→) und Zeilenenden (¶) nativ.
2. NonPrintableHighlighter → alle anderen Steuerzeichen erhalten einen farbigen Hintergrund,
   sodass sie als farbige Kästchen sichtbar werden.
"""

import re

from PyQt6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat, QTextOption
from PyQt6.QtWidgets import QPlainTextEdit

# Steuerzeichen ohne \t (0x09) und \n (0x0A) — Qt behandelt diese selbst
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")


class NonPrintableHighlighter(QSyntaxHighlighter):
    """Hebt Steuerzeichen mit orangem Hintergrund hervor."""

    def __init__(self, document):
        super().__init__(document)
        self._fmt = QTextCharFormat()
        self._fmt.setBackground(QColor("#e06000"))
        self._fmt.setForeground(QColor("#ffffff"))

    def highlightBlock(self, text: str) -> None:
        for m in _CTRL_RE.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt)


def apply_show_invisibles(editor: QPlainTextEdit, enabled: bool) -> None:
    """
    Schaltet die Anzeige nicht druckbarer Zeichen in einem QPlainTextEdit ein oder aus.
    Gibt das Highlighter-Objekt zurück (oder None), damit der Aufrufer
    eine Referenz halten kann (GC-Schutz).
    """
    doc = editor.document()
    opt = doc.defaultTextOption()

    if enabled:
        opt.setFlags(
            QTextOption.Flag.ShowTabsAndSpaces
            | QTextOption.Flag.ShowLineAndParagraphSeparators
        )
    else:
        opt.setFlags(QTextOption.Flag(0))

    doc.setDefaultTextOption(opt)
    # Neu zeichnen erzwingen
    doc.markContentsDirty(0, doc.characterCount())
