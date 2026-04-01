import json
import re

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# LLM-Korrektur-Worker
# ---------------------------------------------------------------------------

class LlmCorrectionWorker(QThread):
    finished = pyqtSignal(list)   # list of {"original": ..., "korrektur": ...}
    error = pyqtSignal(str)

    def __init__(self, fundstellen: list[dict], ollama_client, model: str):
        super().__init__()
        self._fundstellen = fundstellen
        self._client = ollama_client
        self._model = model

    def run(self):
        items = "\n".join(
            f"{i + 1}. {f.get('raw_kontext', f.get('kontext', ''))}"
            for i, f in enumerate(self._fundstellen)
        )
        system = (
            "Du bist ein Korrekturassistent für OCR-Fehler in deutschen Texten. "
            "Korrigiere die genannten Textstellen (besonders fehlerhafte Anführungszeichen "
            "und ungewöhnliche Zeichen). "
            "Antworte NUR mit einem JSON-Array im Format: "
            '[{"original": "...", "korrektur": "..."}]. '
            "Keine weiteren Erklärungen."
        )
        user = f"Korrigiere diese Textstellen:\n{items}"
        try:
            result = self._client.generate(self._model, system, user)
            match = re.search(r"\[.*?\]", result, re.DOTALL)
            if match:
                corrections = json.loads(match.group())
                self.finished.emit(corrections)
            else:
                self.error.emit("LLM gab kein gültiges JSON zurück.")
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Widget für eine einzelne Fundstelle
# ---------------------------------------------------------------------------

class FundstelleWidget(QFrame):
    ignored = pyqtSignal(object)              # self
    corrected = pyqtSignal(object, str, str)  # self, old_raw, new_raw

    _TYP_LABELS = {
        "unpaarige_anfuehrungszeichen": "Unpaarige Anführungszeichen",
        "verdaechtige_zeichen": "Verdächtige Zeichen",
        "ungewoehnliche_kombinationen": "Ungewöhnliche Kombination",
    }

    def __init__(self, fundstelle: dict, parent=None):
        super().__init__(parent)
        self._fundstelle = fundstelle
        self._done = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        typ_display = self._TYP_LABELS.get(
            self._fundstelle["typ"], self._fundstelle["typ"]
        )
        header = QLabel(
            f"<b>Zeile {self._fundstelle['zeile']}</b>"
            f" | {typ_display}"
        )
        layout.addWidget(header)

        kontext_html = self._fundstelle["kontext"].replace("<", "&lt;").replace(">", "&gt;")
        self._context_label = QLabel(f"Kontext: <code>{kontext_html}</code>")
        self._context_label.setWordWrap(True)
        layout.addWidget(self._context_label)

        # Inline-Edit-Bereich (anfangs ausgeblendet)
        self._edit_widget = QWidget()
        edit_layout = QVBoxLayout(self._edit_widget)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(4)

        self._edit_field = QLineEdit()
        self._edit_field.setText(
            self._fundstelle.get("raw_kontext", self._fundstelle.get("kontext", ""))
        )
        edit_layout.addWidget(self._edit_field)

        edit_btn_row = QHBoxLayout()
        confirm_btn = QPushButton("✓ Bestätigen")
        confirm_btn.clicked.connect(self._confirm_edit)
        edit_btn_row.addWidget(confirm_btn)

        cancel_btn = QPushButton("✗ Abbrechen")
        cancel_btn.clicked.connect(self._cancel_edit)
        edit_btn_row.addWidget(cancel_btn)
        edit_btn_row.addStretch()
        edit_layout.addLayout(edit_btn_row)

        self._edit_widget.setVisible(False)
        layout.addWidget(self._edit_widget)

        # Aktions-Buttons
        self._btn_widget = QWidget()
        btn_layout = QHBoxLayout(self._btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        ignore_btn = QPushButton("Ignorieren")
        ignore_btn.clicked.connect(self._on_ignore)
        btn_layout.addWidget(ignore_btn)

        edit_btn = QPushButton("✎ Manuell bearbeiten")
        edit_btn.clicked.connect(self._start_edit)
        btn_layout.addWidget(edit_btn)

        btn_layout.addStretch()
        layout.addWidget(self._btn_widget)

        # Status-Label (anfangs ausgeblendet)
        self._done_label = QLabel("✓ Erledigt")
        self._done_label.setStyleSheet("color: #888;")
        self._done_label.setVisible(False)
        layout.addWidget(self._done_label)

    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self._done

    def set_llm_suggestion(self, suggestion: str):
        """Füllt das Editierfeld mit einem LLM-Vorschlag und öffnet es."""
        self._edit_field.setText(suggestion)
        self._start_edit()

    def mark_done(self):
        self._done = True
        self._btn_widget.setVisible(False)
        self._edit_widget.setVisible(False)
        self._done_label.setVisible(True)
        self.setStyleSheet(
            "FundstelleWidget { background: #f4f4f4; } "
            "QLabel { color: #aaa; }"
        )

    # ------------------------------------------------------------------

    def _on_ignore(self):
        self.mark_done()
        self.ignored.emit(self)

    def _start_edit(self):
        self._btn_widget.setVisible(False)
        self._edit_widget.setVisible(True)
        self._edit_field.setFocus()
        self._edit_field.selectAll()

    def _cancel_edit(self):
        self._edit_widget.setVisible(False)
        self._btn_widget.setVisible(True)

    def _confirm_edit(self):
        old = self._fundstelle.get("raw_kontext", "")
        new = self._edit_field.text()
        if old != new:
            self.corrected.emit(self, old, new)
        self.mark_done()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class OcrReviewDialog(QDialog):
    def __init__(
        self,
        scan_result: dict,
        main_text: str,
        ollama_client,
        model: str,
        parent=None,
    ):
        super().__init__(parent)
        self._scan_result = scan_result
        self._current_text = main_text
        self._ollama_client = ollama_client
        self._model = model
        self._widgets: list[FundstelleWidget] = []
        self._pending_llm_widgets: list[FundstelleWidget] = []
        self._llm_worker: LlmCorrectionWorker | None = None

        self.setWindowTitle("OCR-Fundstellen überprüfen")
        self.resize(720, 600)
        self._setup_ui()
        self._populate()
        self._update_progress()

    def get_corrected_text(self) -> str:
        return self._current_text

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Kopfzeile
        top_row = QHBoxLayout()

        self._progress_label = QLabel("Fortschritt: 0/0 erledigt")
        top_row.addWidget(self._progress_label)
        top_row.addStretch()

        ignore_all_btn = QPushButton("Alle ignorieren")
        ignore_all_btn.clicked.connect(self._ignore_all)
        top_row.addWidget(ignore_all_btn)

        self._llm_btn = QPushButton("\U0001f916  LLM-Korrektur für verbleibende")
        self._llm_btn.clicked.connect(self._llm_correct_remaining)
        top_row.addWidget(self._llm_btn)

        layout.addLayout(top_row)

        # Scrollbereich für Fundstellen
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._scroll_layout = QVBoxLayout(self._container)
        self._scroll_layout.setSpacing(6)
        self._scroll_layout.addStretch()

        scroll.setWidget(self._container)
        layout.addWidget(scroll, stretch=1)

        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _populate(self):
        # Stretch am Ende entfernen, Items einfügen, dann neu anhängen
        stretch = self._scroll_layout.takeAt(self._scroll_layout.count() - 1)
        del stretch

        for fundstelle in self._scan_result.get("fundstellen", []):
            w = FundstelleWidget(fundstelle)
            w.ignored.connect(lambda _: self._update_progress())
            w.corrected.connect(self._apply_correction)
            self._widgets.append(w)
            self._scroll_layout.addWidget(w)

        self._scroll_layout.addStretch()

    # ------------------------------------------------------------------
    # Aktionen
    # ------------------------------------------------------------------

    def _update_progress(self):
        done = sum(1 for w in self._widgets if w.is_done())
        total = len(self._widgets)
        self._progress_label.setText(f"Fortschritt: {done}/{total} erledigt")

    def _apply_correction(self, _widget, old: str, new: str):
        if old and old in self._current_text:
            self._current_text = self._current_text.replace(old, new, 1)
        self._update_progress()

    def _ignore_all(self):
        for w in self._widgets:
            if not w.is_done():
                w.mark_done()
        self._update_progress()

    def _llm_correct_remaining(self):
        open_widgets = [w for w in self._widgets if not w.is_done()]
        if not open_widgets:
            QMessageBox.information(
                self, "LLM-Korrektur", "Alle Fundstellen sind bereits erledigt."
            )
            return

        open_indices = [
            i for i, w in enumerate(self._widgets) if not w.is_done()
        ]
        fundstellen = [self._scan_result["fundstellen"][i] for i in open_indices]
        self._pending_llm_widgets = open_widgets

        self._llm_btn.setEnabled(False)
        self._llm_btn.setText("LLM läuft…")

        self._llm_worker = LlmCorrectionWorker(
            fundstellen, self._ollama_client, self._model
        )
        self._llm_worker.finished.connect(self._on_llm_finished)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.start()

    def _on_llm_finished(self, corrections: list):
        self._llm_btn.setEnabled(True)
        self._llm_btn.setText("\U0001f916  LLM-Korrektur für verbleibende")

        for idx, correction in enumerate(corrections):
            if idx < len(self._pending_llm_widgets):
                suggestion = correction.get("korrektur", "")
                if suggestion:
                    self._pending_llm_widgets[idx].set_llm_suggestion(suggestion)

    def _on_llm_error(self, msg: str):
        self._llm_btn.setEnabled(True)
        self._llm_btn.setText("\U0001f916  LLM-Korrektur für verbleibende")
        QMessageBox.warning(self, "LLM-Fehler", f"Fehler bei der LLM-Korrektur:\n{msg}")
