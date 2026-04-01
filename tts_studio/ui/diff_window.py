import difflib

from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class DiffWindow(QDialog):
    def __init__(self, original: str, modified: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diff: Original \u2194 Verarbeitet")
        self.resize(920, 700)
        self._original = original
        self._modified = modified
        self._setup_ui()
        self._render_diff()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        legend = QLabel(
            "<span style='color:#cc0000; background:#ffdddd;'>\u2212 entfernt</span>"
            "&nbsp;&nbsp;&nbsp;"
            "<span style='color:#006600; background:#ddffdd;'>+ hinzugef\u00fcgt</span>"
            "&nbsp;&nbsp;&nbsp;"
            "<span style='color:#444; background:#e8e8e8;'>\u00b7 Kontext</span>"
        )
        legend.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(legend)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(self.text_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Schlie\u00dfen")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _render_diff(self):
        original_lines = self._original.splitlines(keepends=True)
        modified_lines = self._modified.splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile="Original",
                tofile="Verarbeitet",
                lineterm="",
            )
        )

        cursor: QTextCursor = self.text_edit.textCursor()

        fmt_removed = QTextCharFormat()
        fmt_removed.setBackground(QColor("#ffdddd"))
        fmt_removed.setForeground(QColor("#cc0000"))

        fmt_added = QTextCharFormat()
        fmt_added.setBackground(QColor("#ddffdd"))
        fmt_added.setForeground(QColor("#006600"))

        fmt_header = QTextCharFormat()
        fmt_header.setBackground(QColor("#e0e0e0"))
        fmt_header.setForeground(QColor("#333333"))
        fmt_header.setFontWeight(700)

        fmt_hunk = QTextCharFormat()
        fmt_hunk.setBackground(QColor("#ddeeff"))
        fmt_hunk.setForeground(QColor("#0055aa"))

        fmt_normal = QTextCharFormat()

        if not diff:
            cursor.setCharFormat(fmt_normal)
            cursor.insertText("Keine Unterschiede gefunden – Texte sind identisch.")
            return

        for line in diff:
            text = line.rstrip("\n") + "\n"
            if line.startswith("---") or line.startswith("+++"):
                cursor.setCharFormat(fmt_header)
            elif line.startswith("@@"):
                cursor.setCharFormat(fmt_hunk)
            elif line.startswith("-"):
                cursor.setCharFormat(fmt_removed)
            elif line.startswith("+"):
                cursor.setCharFormat(fmt_added)
            else:
                cursor.setCharFormat(fmt_normal)
            cursor.insertText(text)

        self.text_edit.setTextCursor(cursor)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
