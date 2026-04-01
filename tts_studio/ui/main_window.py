import json
import os
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.epub_reader import EpubReader, EpubReaderError
from core.ocr_scanner import OcrScanner
from core.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    preprocess_text,
)
from core.prompt_builder import PromptBuilder
from core.rules_manager import RulesManager
from core.settings_manager import SettingsManager
from core.tts_bridge import TtsBridge
from core.voice_profile_manager import VoiceProfileManager
from ui.diff_window import DiffWindow
from ui.invisible_chars import NonPrintableHighlighter, apply_show_invisibles
from ui.ocr_review_dialog import OcrReviewDialog
from ui.rule_editor_dialog import RuleEditorDialog
from ui.settings_dialog import SettingsDialog


# ---------------------------------------------------------------------------
# Hintergrund-Worker: Text-Korrektur
# ---------------------------------------------------------------------------

class ProcessingWorker(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, status_msg
    finished = pyqtSignal(str)             # combined result text
    error = pyqtSignal(str)               # error message

    def __init__(
        self,
        text: str,
        model: str,
        chunk_size: int,
        ollama_client: OllamaClient,
        prompt_builder: PromptBuilder,
        cache_dir: str = "",
        input_file: str = "",
        start_index: int = 0,
        initial_results: list[str] | None = None,
    ):
        super().__init__()
        self.text = text
        self.model = model
        self.chunk_size = chunk_size
        self.client = ollama_client
        self.builder = prompt_builder
        self._cache_dir = cache_dir
        self._input_file = input_file
        self._start_index = start_index
        self._initial_results: list[str] = list(initial_results or [])
        self._cancelled = False
        self._paused = threading.Event()
        self._paused.set()  # not paused initially

    def cancel(self):
        self._cancelled = True
        self._paused.set()  # unblock if paused so the thread can exit

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            system_prompt = self.builder.build_system_prompt()
            preprocessed = preprocess_text(self.text)
            chunks = self._split_into_chunks(preprocessed, self.chunk_size)
            total = len(chunks)
            results: list[str] = list(self._initial_results)

            for i, chunk in enumerate(
                chunks[self._start_index:], self._start_index + 1
            ):
                if self._cancelled:
                    break
                # Pause-Warteschleife
                while not self._paused.wait(timeout=0.1):
                    if self._cancelled:
                        break
                if self._cancelled:
                    break
                self.progress.emit(i, total, f"Chunk {i}/{total} wird verarbeitet\u2026")
                result = self.client.generate(
                    self.model,
                    system_prompt,
                    chunk,
                    cancelled_callback=self.is_cancelled,
                )
                if self._cancelled:
                    break
                results.append(result)
                self._save_cache(total, i, results)

            if not self._cancelled:
                self._clear_cache()
                self.finished.emit("\n\n".join(results))

        except OllamaConnectionError:
            self.error.emit(
                "Ollama ist nicht erreichbar.\n\n"
                "Bitte starten Sie Ollama mit dem Befehl:\n\n"
                "    ollama serve"
            )
        except OllamaModelNotFoundError as exc:
            model_list = (
                "\n".join(f"  \u2022 {m}" for m in exc.available)
                if exc.available
                else "  (keine Modelle gefunden)"
            )
            self.error.emit(
                f"Modell \u201e{exc.model}\u201c nicht gefunden.\n\n"
                f"Verf\u00fcgbare Modelle:\n{model_list}"
            )
        except Exception as exc:
            self.error.emit(f"Fehler bei der Verarbeitung:\n{exc}")
        finally:
            try:
                self.client.unload_model(self.model)
            except Exception:
                pass

    # ------------------------------------------------------------------

    def _cache_file_path(self) -> str | None:
        if not self._cache_dir or not self._input_file:
            return None
        stem = Path(self._input_file).stem
        return str(Path(self._cache_dir) / f"{stem}_cache.json")

    def _save_cache(self, total: int, done: int, results: list[str]):
        path = self._cache_file_path()
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "eingabe_datei": self._input_file,
            "gesamt_chunks": total,
            "verarbeitete_chunks": done,
            "ergebnisse": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _clear_cache(self):
        path = self._cache_file_path()
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _split_into_chunks(text: str, max_size: int) -> list[str]:
        if not text.strip():
            return []
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_parts: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            sep_size = 2 if current_parts else 0
            if current_parts and current_size + sep_size + para_size > max_size:
                chunks.append("\n\n".join(current_parts))
                current_parts = [para]
                current_size = para_size
            else:
                current_parts.append(para)
                current_size += sep_size + para_size

        if current_parts:
            chunks.append("\n\n".join(current_parts))
        return chunks or [text]


# ---------------------------------------------------------------------------
# Hintergrund-Worker: TTS
# ---------------------------------------------------------------------------

class TtsWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        bridge: TtsBridge,
        text: str,
        filename: str,
        formats: list[str],
    ):
        super().__init__()
        self._bridge = bridge
        self._text = text
        self._filename = filename
        self._formats = formats

    def run(self):
        result = self._bridge.run(
            self._text,
            self._filename,
            self._formats,
            progress_callback=lambda msg: self.progress.emit(msg),
        )
        if result["success"]:
            self.finished.emit(result)
        else:
            self.error.emit(result.get("error", "Unbekannter Fehler"))


# ---------------------------------------------------------------------------
# Hintergrund-Worker: Vorspann-Erkennung
# ---------------------------------------------------------------------------

class VorspannWorker(QThread):
    detected = pyqtSignal(str)   # erste Zeile des eigentlichen Inhalts (leer = kein Vorspann)
    error = pyqtSignal(str)

    _SYSTEM_PROMPT = (
        "Du analysierst den Beginn eines deutschen Buches. "
        "Der Text enthält möglicherweise einen Vorspann (Impressum, Widmung, "
        "Inhaltsverzeichnis, Vorwort usw.) bevor der eigentliche Inhalt beginnt. "
        "Antworte NUR mit der ersten Zeile des eigentlichen Inhalts, "
        "ohne Anführungszeichen oder Erklärungen. "
        "Wenn kein Vorspann erkennbar ist oder du dir nicht sicher bist, "
        "antworte mit einer leeren Antwort."
    )

    def __init__(self, client: OllamaClient, model: str, text: str):
        super().__init__()
        self._client = client
        self._model = model
        self._text = text[:3000]

    def run(self):
        try:
            result = self._client.generate(self._model, self._SYSTEM_PROMPT, self._text)
            self.detected.emit(result.strip())
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Hauptfenster
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Studio")
        self.setWindowIcon(QIcon(r"D:\Downloads\favicon(1).ico"))

        self.settings = SettingsManager()
        self.rules_manager = RulesManager()
        self.prompt_builder = PromptBuilder(self.rules_manager)
        self.ollama_client = OllamaClient(
            self.settings.get("ollama_url", "http://localhost:11434")
        )
        self._ocr_scanner = OcrScanner()
        self._voice_profile_manager = VoiceProfileManager()

        self._worker: ProcessingWorker | None = None
        self._tts_worker: TtsWorker | None = None
        self._vorspann_worker: VorspannWorker | None = None
        self._input_file: str | None = None
        self._tts_input_file: str | None = None
        self._last_scan_result: dict | None = None
        self._show_invisibles: bool = False
        self._hl_input: NonPrintableHighlighter | None = None
        self._hl_output: NonPrintableHighlighter | None = None
        # Marker-State
        self._marker_start: int | None = None
        self._marker_end: int | None = None
        # Verarbeitungs-/TTS-Workflow-State
        self._output_up_to_date: bool = False
        self._pending_tts_after_processing: bool = False
        self._ocr_findings_reviewed: bool = False

        self._setup_ui()
        self._setup_menu()
        self._refresh_rules_list()
        self._refresh_voice_profiles()

        QTimer.singleShot(200, self._load_available_models)

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------

    def _setup_ui(self):
        self.setMinimumSize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        root_layout.addWidget(self.tab_widget)

        self.tab_widget.addTab(self._build_text_correction_tab(), "Textkorrektur")
        self.tab_widget.addTab(self._build_tts_tab(), "TTS-Ausgabe")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Bereit")

    def _build_text_correction_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_text_correction_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([680, 320])
        return splitter

    def _build_text_correction_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 6, 8)
        layout.setSpacing(6)

        # ── OCR-Analyse ───────────────────────────────────────────────
        ocr_group = QGroupBox("OCR-Analyse")
        ocr_layout = QHBoxLayout(ocr_group)
        ocr_layout.setSpacing(8)

        self.ocr_scan_btn = QPushButton("\U0001f4cb  OCR-Scan starten")
        self.ocr_scan_btn.setStyleSheet(
            "QPushButton { padding: 8px 20px; border-radius: 4px; }"
        )
        self.ocr_scan_btn.setToolTip(
            "Analysiert den Eingabetext auf OCR-Fehler wie unpaarige "
            "Anf\u00fchrungszeichen und verd\u00e4chtige Zeichen"
        )
        self.ocr_scan_btn.clicked.connect(self._run_ocr_scan)
        ocr_layout.addWidget(self.ocr_scan_btn)

        self.ocr_quality_label = QLabel("")
        self.ocr_quality_label.setVisible(False)
        ocr_layout.addWidget(self.ocr_quality_label)

        self.ocr_detail_btn = QPushButton("")
        self.ocr_detail_btn.setFlat(True)
        self.ocr_detail_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ocr_detail_btn.setStyleSheet(
            "QPushButton { color: #0055aa; }"
            "QPushButton:hover { color: #003399; }"
        )
        self.ocr_detail_btn.setVisible(False)
        self.ocr_detail_btn.clicked.connect(self._show_ocr_fundstellen)
        ocr_layout.addWidget(self.ocr_detail_btn)

        ocr_layout.addStretch()
        layout.addWidget(ocr_group)

        # ── Eingabe ───────────────────────────────────────────────────
        input_group = QGroupBox("Eingabe")
        input_layout = QVBoxLayout(input_group)
        input_layout.setContentsMargins(6, 6, 6, 6)
        input_layout.setSpacing(4)

        marker_row = QHBoxLayout()
        marker_row.setSpacing(4)

        self._btn_start_marker = QPushButton("\u2702 Start hier")
        self._btn_start_marker.setToolTip(
            "Setzt den Startmarker an die aktuelle Cursor-Position.\n"
            "Text vor dem Marker wird ausgegraut (Vorspann)."
        )
        self._btn_start_marker.clicked.connect(self._set_start_marker)
        marker_row.addWidget(self._btn_start_marker)

        self._btn_end_marker = QPushButton("\u2702 Ende hier")
        self._btn_end_marker.setToolTip(
            "Setzt den Endmarker an die aktuelle Cursor-Position.\n"
            "Text nach dem Marker wird ausgegraut (Nachspann)."
        )
        self._btn_end_marker.clicked.connect(self._set_end_marker)
        marker_row.addWidget(self._btn_end_marker)

        self._btn_reset_markers = QPushButton("\u21ba Marker zur\u00fccksetzen")
        self._btn_reset_markers.setToolTip("Entfernt beide Marker und zeigt den gesamten Text.")
        self._btn_reset_markers.clicked.connect(self._reset_markers)
        marker_row.addWidget(self._btn_reset_markers)

        self._btn_detect_preamble = QPushButton("\U0001f916 Vorspann automatisch erkennen")
        self._btn_detect_preamble.setToolTip(
            "Sendet die ersten 3000 Zeichen an das LLM,\n"
            "um den Vorspann automatisch zu erkennen und den Startmarker zu setzen."
        )
        self._btn_detect_preamble.clicked.connect(self._detect_preamble)
        marker_row.addWidget(self._btn_detect_preamble)

        marker_row.addStretch()
        input_layout.addLayout(marker_row)

        self.input_text = QPlainTextEdit()
        self.input_text.setPlaceholderText(
            "Text hier eingeben oder \u00fcber Datei \u203a \u00d6ffnen laden\u2026"
        )
        self.input_text.setFont(QFont("Segoe UI", 10))
        input_layout.addWidget(self.input_text)

        layout.addWidget(input_group, stretch=1)

        # ── Verarbeitung ──────────────────────────────────────────────
        proc_group = QGroupBox("Verarbeitung")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.process_btn = QPushButton("\u25b6  Verarbeiten")
        self.process_btn.setStyleSheet(
            "QPushButton {"
            "  background:#2c7a2c; color:white; padding:6px 18px;"
            "  border-radius:4px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#389438; }"
            "QPushButton:disabled { background:#999; color:#ccc; }"
        )
        self.process_btn.setToolTip(
            "Sendet den Text chunk-weise an das gew\u00e4hlte "
            "Ollama-Modell zur Vorverarbeitung"
        )
        self.process_btn.clicked.connect(self._start_processing)
        btn_row.addWidget(self.process_btn)

        self.cancel_btn = QPushButton("\u2715  Abbrechen")
        self.cancel_btn.setStyleSheet(
            "QPushButton {"
            "  background:#aa2222; color:white; padding:6px 14px;"
            "  border-radius:4px;"
            "}"
            "QPushButton:hover { background:#cc3333; }"
        )
        self.cancel_btn.setToolTip(
            "Bricht die laufende Verarbeitung nach dem aktuellen Chunk ab"
        )
        self.cancel_btn.clicked.connect(self._cancel_processing)
        self.cancel_btn.setVisible(False)
        btn_row.addWidget(self.cancel_btn)

        self.pause_btn = QPushButton("\u23f8  Pause")
        self.pause_btn.setStyleSheet(
            "QPushButton {"
            "  background:#5566aa; color:white; padding:6px 14px;"
            "  border-radius:4px;"
            "}"
            "QPushButton:hover { background:#6677bb; }"
        )
        self.pause_btn.setToolTip(
            "H\u00e4lt die Verarbeitung nach dem aktuellen Chunk an "
            "\u2013 kann fortgesetzt werden"
        )
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setVisible(False)
        btn_row.addWidget(self.pause_btn)

        self.diff_btn = QPushButton("\u2195  Diff anzeigen")
        self.diff_btn.setToolTip(
            "Zeigt einen Vergleich zwischen Original und verarbeitetem Text"
        )
        self.diff_btn.clicked.connect(self._show_diff)
        btn_row.addWidget(self.diff_btn)

        self.invisibles_btn = QPushButton("\u00b6  Sonderzeichen")
        self.invisibles_btn.setCheckable(True)
        self.invisibles_btn.setToolTip(
            "Blendet nicht druckbare Zeichen wie Leerzeichen, Tabs und Zeilenenden ein"
        )
        self.invisibles_btn.toggled.connect(self._toggle_invisibles)
        btn_row.addWidget(self.invisibles_btn)

        btn_row.addStretch()
        proc_layout.addLayout(btn_row)

        self.chunk_progress_bar = QProgressBar()
        self.chunk_progress_bar.setRange(0, 1)
        self.chunk_progress_bar.setValue(0)
        self.chunk_progress_bar.setTextVisible(True)
        self.chunk_progress_bar.setFormat("Chunk %v / %m")
        self.chunk_progress_bar.setFixedHeight(14)
        self.chunk_progress_bar.setVisible(False)
        proc_layout.addWidget(self.chunk_progress_bar)

        layout.addWidget(proc_group)

        # ── Ausgabe ───────────────────────────────────────────────────
        output_group = QGroupBox("Ausgabe")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(6, 6, 6, 6)
        output_layout.setSpacing(4)

        self.output_text = QPlainTextEdit()
        self.output_text.setPlaceholderText("Verarbeiteter Text erscheint hier\u2026")
        self.output_text.setFont(QFont("Segoe UI", 10))
        output_layout.addWidget(self.output_text)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_btn = QPushButton("\U0001f4be  Speichern")
        self.save_btn.setToolTip("Speichert den verarbeiteten Text als Datei")
        self.save_btn.clicked.connect(self._save_output)
        save_row.addWidget(self.save_btn)
        output_layout.addLayout(save_row)

        layout.addWidget(output_group, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 8)
        layout.setSpacing(6)

        # ── Regeln ────────────────────────────────────────────────────
        rules_group = QGroupBox("Regeln")
        rules_layout = QVBoxLayout(rules_group)
        rules_layout.setContentsMargins(6, 6, 6, 6)
        rules_layout.setSpacing(4)

        self.rules_list = QListWidget()
        self.rules_list.setAlternatingRowColors(True)
        self.rules_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.rules_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.rules_list.itemDoubleClicked.connect(self._edit_rule)
        self.rules_list.itemChanged.connect(self._on_rule_check_changed)
        rules_layout.addWidget(self.rules_list, stretch=1)

        crud_row = QHBoxLayout()
        crud_row.setSpacing(4)

        btn_new = QPushButton("+ Neu")
        btn_new.clicked.connect(self._new_rule)
        crud_row.addWidget(btn_new)

        btn_edit = QPushButton("\u270e Bearbeiten")
        btn_edit.clicked.connect(self._edit_rule)
        crud_row.addWidget(btn_edit)

        btn_del = QPushButton("\U0001f5d1 L\u00f6schen")
        btn_del.setStyleSheet("color: #aa2222;")
        btn_del.clicked.connect(self._delete_rule)
        crud_row.addWidget(btn_del)

        rules_layout.addLayout(crud_row)

        prio_row = QHBoxLayout()
        prio_row.setSpacing(4)

        btn_up = QPushButton("\u2191 H\u00f6her")
        btn_up.clicked.connect(self._move_rule_up)
        prio_row.addWidget(btn_up)

        btn_down = QPushButton("\u2193 Niedriger")
        btn_down.clicked.connect(self._move_rule_down)
        prio_row.addWidget(btn_down)

        rules_layout.addLayout(prio_row)

        layout.addWidget(rules_group, stretch=1)

        # ── Modell & Chunk-Größe ──────────────────────────────────────
        model_group = QGroupBox("Modell")
        model_group_layout = QVBoxLayout(model_group)
        model_group_layout.setContentsMargins(6, 6, 6, 6)
        model_group_layout.setSpacing(6)

        model_row = QHBoxLayout()
        model_row.setSpacing(4)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        default_model = self.settings.get("default_model", "gemma3:12b")
        self.model_combo.addItem(default_model)
        self.model_combo.setCurrentText(default_model)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo, stretch=1)

        btn_refresh_models = QPushButton("\u21bb")
        btn_refresh_models.setFixedWidth(28)
        btn_refresh_models.setToolTip("Verf\u00fcgbare Modelle neu laden")
        btn_refresh_models.clicked.connect(self._load_available_models)
        model_row.addWidget(btn_refresh_models)

        model_group_layout.addLayout(model_row)

        lbl_chunk = QLabel("Chunk-Gr\u00f6\u00dfe:")
        model_group_layout.addWidget(lbl_chunk)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(500, 10000)
        self.chunk_spin.setSingleStep(500)
        self.chunk_spin.setSuffix(" Zeichen")
        self.chunk_spin.setValue(self.settings.get("default_chunk_size", 2500))
        self.chunk_spin.valueChanged.connect(
            lambda v: self.settings.set("default_chunk_size", v)
        )
        model_group_layout.addWidget(self.chunk_spin)

        layout.addWidget(model_group)

        return panel

    # ------------------------------------------------------------------
    # TTS-Tab
    # ------------------------------------------------------------------

    def _build_tts_tab(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        outer.addWidget(self._build_tts_text_group())
        outer.addWidget(self._build_tts_voice_group())
        outer.addWidget(self._build_tts_output_group())
        outer.addWidget(self._build_tts_action_group())
        outer.addStretch()
        return container

    def _build_tts_text_group(self) -> QGroupBox:
        group = QGroupBox("Texteingabe")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self._tts_radio_tab1 = QRadioButton(
            "Verarbeiteten Text aus Tab\u00a01 \u00fcbernehmen"
        )
        self._tts_radio_tab1.setChecked(True)
        self._tts_radio_file = QRadioButton("Datei laden\u2026")

        bg = QButtonGroup(self)
        bg.addButton(self._tts_radio_tab1)
        bg.addButton(self._tts_radio_file)
        bg.buttonClicked.connect(self._tts_text_source_changed)

        layout.addWidget(self._tts_radio_tab1)

        file_row = QHBoxLayout()
        file_row.addWidget(self._tts_radio_file)
        self._tts_file_edit = QLineEdit()
        self._tts_file_edit.setPlaceholderText("Pfad zur Textdatei\u2026")
        self._tts_file_edit.setReadOnly(True)
        file_row.addWidget(self._tts_file_edit, stretch=1)
        browse_file_btn = QPushButton("Durchsuchen\u2026")
        browse_file_btn.clicked.connect(self._browse_tts_input_file)
        file_row.addWidget(browse_file_btn)
        layout.addLayout(file_row)

        self._tts_preview = QPlainTextEdit()
        self._tts_preview.setReadOnly(True)
        self._tts_preview.setMaximumHeight(90)
        self._tts_preview.setPlaceholderText("Textvorschau\u2026")
        self._tts_preview.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self._tts_preview)

        return group

    def _build_tts_voice_group(self) -> QGroupBox:
        group = QGroupBox("Stimmprofil")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(6)
        profile_row.addWidget(QLabel("Stimmprofil:"))

        self._voice_combo = QComboBox()
        profile_row.addWidget(self._voice_combo, stretch=1)

        btn_new_profile = QPushButton("+ Neu")
        btn_new_profile.clicked.connect(self._show_new_profile_form)
        profile_row.addWidget(btn_new_profile)

        btn_del_profile = QPushButton("\U0001f5d1 L\u00f6schen")
        btn_del_profile.setStyleSheet("color: #aa2222;")
        btn_del_profile.clicked.connect(self._delete_voice_profile)
        profile_row.addWidget(btn_del_profile)

        layout.addLayout(profile_row)

        # Formular für neues Profil
        self._profile_form = QWidget()
        form_layout = QVBoxLayout(self._profile_form)
        form_layout.setContentsMargins(0, 4, 0, 0)
        form_layout.setSpacing(6)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._profile_name_edit = QLineEdit()
        self._profile_name_edit.setPlaceholderText("Profilname (Pflichtfeld)")
        name_row.addWidget(self._profile_name_edit)
        form_layout.addLayout(name_row)

        wav_row = QHBoxLayout()
        wav_row.addWidget(QLabel("Referenz-WAV:"))
        self._profile_wav_edit = QLineEdit()
        self._profile_wav_edit.setPlaceholderText("Pfad zur Referenz-WAV\u2026")
        wav_row.addWidget(self._profile_wav_edit, stretch=1)
        btn_browse_wav = QPushButton("Durchsuchen\u2026")
        btn_browse_wav.clicked.connect(self._browse_profile_wav)
        wav_row.addWidget(btn_browse_wav)
        form_layout.addLayout(wav_row)

        txt_row = QHBoxLayout()
        txt_row.addWidget(QLabel("Referenz-TXT:"))
        self._profile_txt_edit = QLineEdit()
        self._profile_txt_edit.setPlaceholderText("Pfad zur Referenz-TXT\u2026")
        txt_row.addWidget(self._profile_txt_edit, stretch=1)
        btn_browse_txt = QPushButton("Durchsuchen\u2026")
        btn_browse_txt.clicked.connect(self._browse_profile_txt)
        txt_row.addWidget(btn_browse_txt)
        form_layout.addLayout(txt_row)

        save_row = QHBoxLayout()
        save_row.addStretch()
        btn_save_profile = QPushButton("Profil speichern")
        btn_save_profile.clicked.connect(self._save_voice_profile)
        save_row.addWidget(btn_save_profile)
        btn_cancel_profile = QPushButton("Abbrechen")
        btn_cancel_profile.clicked.connect(
            lambda: self._profile_form.setVisible(False)
        )
        save_row.addWidget(btn_cancel_profile)
        form_layout.addLayout(save_row)

        self._profile_form.setVisible(False)
        layout.addWidget(self._profile_form)

        return group

    def _build_tts_output_group(self) -> QGroupBox:
        group = QGroupBox("Ausgabe")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Ausgabeordner:"))
        self._tts_output_dir_edit = QLineEdit()
        self._tts_output_dir_edit.setText(self.settings.get("tts_output_dir", ""))
        dir_row.addWidget(self._tts_output_dir_edit, stretch=1)
        btn_browse_dir = QPushButton("Durchsuchen\u2026")
        btn_browse_dir.clicked.connect(self._browse_tts_output_dir)
        dir_row.addWidget(btn_browse_dir)
        layout.addLayout(dir_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Ausgabeformate:"))
        saved_formats = self.settings.get("output_formats", ["mp3"])
        self._fmt_mp3 = QCheckBox("MP3")
        self._fmt_mp3.setChecked("mp3" in saved_formats)
        self._fmt_m4b = QCheckBox("M4B")
        self._fmt_m4b.setChecked("m4b" in saved_formats)
        self._fmt_wav = QCheckBox("WAV")
        self._fmt_wav.setChecked("wav" in saved_formats)
        fmt_row.addWidget(self._fmt_mp3)
        fmt_row.addWidget(self._fmt_m4b)
        fmt_row.addWidget(self._fmt_wav)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Dateiname:"))
        self._tts_filename_edit = QLineEdit()
        self._tts_filename_edit.setPlaceholderText("Dateiname ohne Endung\u2026")
        name_row.addWidget(self._tts_filename_edit, stretch=1)
        layout.addLayout(name_row)

        return group

    def _build_tts_action_group(self) -> QGroupBox:
        group = QGroupBox("Aktionen")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        btn_row = QHBoxLayout()
        self._tts_start_btn = QPushButton("\u25b6  TTS starten")
        self._tts_start_btn.setStyleSheet(
            "QPushButton {"
            "  background:#2c7a2c; color:white; padding:6px 20px;"
            "  border-radius:4px; font-weight:bold;"
            "}"
            "QPushButton:hover { background:#389438; }"
            "QPushButton:disabled { background:#999; color:#ccc; }"
        )
        self._tts_start_btn.clicked.connect(self._start_tts)
        btn_row.addWidget(self._tts_start_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        progress_row = QHBoxLayout()
        self._tts_progress_bar = QProgressBar()
        self._tts_progress_bar.setRange(0, 100)
        self._tts_progress_bar.setValue(0)
        self._tts_progress_bar.setVisible(False)
        progress_row.addWidget(self._tts_progress_bar, stretch=1)

        self._tts_progress_label = QLabel("")
        self._tts_progress_label.setVisible(False)
        progress_row.addWidget(self._tts_progress_label)
        layout.addLayout(progress_row)

        return group

    # ------------------------------------------------------------------
    # Menüleiste
    # ------------------------------------------------------------------

    def _setup_menu(self):
        menubar = self.menuBar()

        datei = menubar.addMenu("Datei")

        act_open = QAction("\u00d6ffnen\u2026", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_file)
        datei.addAction(act_open)

        act_save = QAction("Ausgabe speichern\u2026", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_output)
        datei.addAction(act_save)

        datei.addSeparator()

        self.recent_menu = datei.addMenu("Zuletzt verwendet")
        self._update_recent_menu()

        datei.addSeparator()

        act_quit = QAction("Beenden", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        datei.addAction(act_quit)

        settings_menu = menubar.addMenu("Einstellungen")
        act_settings = QAction("Einstellungen\u2026", self)
        act_settings.triggered.connect(self._open_settings)
        settings_menu.addAction(act_settings)

        help_menu = menubar.addMenu("Hilfe")
        act_about = QAction("\u00dcber TTS Studio", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Tab-Verwaltung
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int):
        if index == 1:
            self._update_tts_preview()
            self._check_processing_needed()

    # ------------------------------------------------------------------
    # Regelliste
    # ------------------------------------------------------------------

    def _refresh_rules_list(self):
        self.rules_list.blockSignals(True)
        self.rules_list.clear()

        rules = sorted(
            self.rules_manager.get_all(),
            key=lambda r: r.get("prioritaet", 9999),
        )

        for rule in rules:
            item = QListWidgetItem(rule["name"])
            item.setData(Qt.ItemDataRole.UserRole, rule["id"])
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if rule.get("aktiv", True)
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(
                f"Kategorie: {rule.get('kategorie', '')}\n"
                f"Priorit\u00e4t: {rule.get('prioritaet', '')}"
            )
            self.rules_list.addItem(item)

        self.rules_list.blockSignals(False)

    def _on_rule_check_changed(self, item: QListWidgetItem):
        rule_id = item.data(Qt.ItemDataRole.UserRole)
        rule = self.rules_manager.get_by_id(rule_id)
        if rule:
            rule["aktiv"] = item.checkState() == Qt.CheckState.Checked
            self.rules_manager.update(rule)

    def _new_rule(self):
        dlg = RuleEditorDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            rule = dlg.get_rule()
            if rule:
                self.rules_manager.add(rule)
                self._refresh_rules_list()

    def _edit_rule(self, item: QListWidgetItem | None = None):
        if item is None:
            item = self.rules_list.currentItem()
        if not item:
            return
        rule_id = item.data(Qt.ItemDataRole.UserRole)
        rule = self.rules_manager.get_by_id(rule_id)
        if not rule:
            return
        dlg = RuleEditorDialog(rule=rule, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            updated = dlg.get_rule()
            if updated:
                self.rules_manager.update(updated)
                self._refresh_rules_list()
                self._reselect_rule(rule_id)

    def _delete_rule(self):
        item = self.rules_list.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(
            self,
            "Regel l\u00f6schen",
            f'Regel \u201e{name}\u201c wirklich l\u00f6schen?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            rule_id = item.data(Qt.ItemDataRole.UserRole)
            self.rules_manager.delete(rule_id)
            self._refresh_rules_list()

    def _move_rule_up(self):
        item = self.rules_list.currentItem()
        if not item:
            return
        rule_id = item.data(Qt.ItemDataRole.UserRole)
        if self.rules_manager.move_up(rule_id):
            self._refresh_rules_list()
            self._reselect_rule(rule_id)

    def _move_rule_down(self):
        item = self.rules_list.currentItem()
        if not item:
            return
        rule_id = item.data(Qt.ItemDataRole.UserRole)
        if self.rules_manager.move_down(rule_id):
            self._refresh_rules_list()
            self._reselect_rule(rule_id)

    def _reselect_rule(self, rule_id: str):
        for i in range(self.rules_list.count()):
            it = self.rules_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == rule_id:
                self.rules_list.setCurrentItem(it)
                break

    # ------------------------------------------------------------------
    # Modell-Combo
    # ------------------------------------------------------------------

    def _load_available_models(self):
        models = self.ollama_client.list_models()
        if not models:
            return
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.addItem(current)
            self.model_combo.setCurrentText(current)
        self.model_combo.blockSignals(False)

    def _on_model_changed(self, model: str):
        if model.strip():
            self.settings.set("default_model", model.strip())

    # ------------------------------------------------------------------
    # Text-Korrektur: Verarbeitung
    # ------------------------------------------------------------------

    def _start_processing(self):
        text = self._get_effective_text().strip()
        if not text:
            QMessageBox.warning(self, "Kein Text", "Bitte geben Sie einen Text ein.")
            return
        model = self.model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(self, "Kein Modell", "Bitte w\u00e4hlen Sie ein Modell aus.")
            return

        # Cache prüfen
        start_index = 0
        initial_results: list[str] = []
        cache_dir = self.settings.get("cache_dir", "")
        input_file = self._input_file or ""

        if cache_dir and input_file:
            cache_path = Path(cache_dir) / f"{Path(input_file).stem}_cache.json"
            if cache_path.exists():
                try:
                    with open(cache_path, encoding="utf-8") as f:
                        cache_data = json.load(f)
                    done = cache_data.get("verarbeitete_chunks", 0)
                    total = cache_data.get("gesamt_chunks", 0)
                    reply = QMessageBox.question(
                        self,
                        "Fortschritt gefunden",
                        f"Zwischenspeicher gefunden: {done}/{total} Chunks bereits verarbeitet.\n\n"
                        "Verarbeitung fortsetzen?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        start_index = done
                        initial_results = cache_data.get("ergebnisse", [])
                except Exception:
                    pass

        self.output_text.clear()
        self.process_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.pause_btn.setVisible(True)
        self.pause_btn.setText("\u23f8  Pause")
        self.chunk_progress_bar.setValue(0)
        self.chunk_progress_bar.setRange(0, 1)
        self.chunk_progress_bar.setVisible(True)
        self.status_bar.showMessage("Starte Verarbeitung\u2026")

        self._worker = ProcessingWorker(
            text=text,
            model=model,
            chunk_size=self.chunk_spin.value(),
            ollama_client=self.ollama_client,
            prompt_builder=self.prompt_builder,
            cache_dir=cache_dir,
            input_file=input_file,
            start_index=start_index,
            initial_results=initial_results,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.error.connect(self._on_processing_error)
        self._worker.start()

    def _cancel_processing(self):
        if self._worker:
            self._worker.cancel()
            self.pause_btn.setVisible(False)
            self.status_bar.showMessage("Verarbeitung wird abgebrochen\u2026")

    def _toggle_pause(self):
        if not self._worker:
            return
        if self._worker.is_paused():
            self._worker.resume()
            self.pause_btn.setText("\u23f8  Pause")
            self.status_bar.showMessage("Verarbeitung fortgesetzt.")
        else:
            self._worker.pause()
            self.pause_btn.setText("\u25b6  Fortsetzen")
            self.status_bar.showMessage("Verarbeitung pausiert.")

    def _on_progress(self, current: int, total: int, message: str):
        self.status_bar.showMessage(message)
        self.chunk_progress_bar.setRange(0, total)
        self.chunk_progress_bar.setValue(current)

    def _on_processing_finished(self, result: str):
        self.output_text.setPlainText(result)
        self._output_up_to_date = True
        self.process_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.pause_btn.setVisible(False)
        self.chunk_progress_bar.setVisible(False)
        self.status_bar.showMessage("Verarbeitung abgeschlossen.")
        self._worker = None

        if self._pending_tts_after_processing:
            self._pending_tts_after_processing = False
            if self._has_open_ocr_findings():
                QMessageBox.information(
                    self,
                    "Verarbeitung abgeschlossen",
                    "Die Verarbeitung ist abgeschlossen, aber es liegen noch "
                    "ungepr\u00fcfte OCR-Fundstellen vor.\n\n"
                    "Bitte pr\u00fcfen Sie die Fundstellen, bevor Sie TTS starten.",
                )
            else:
                self.tab_widget.setCurrentIndex(1)
                self._start_tts()

    def _on_processing_error(self, message: str):
        self.process_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.pause_btn.setVisible(False)
        self.chunk_progress_bar.setVisible(False)
        self.status_bar.showMessage("Fehler bei der Verarbeitung.")
        self._worker = None
        QMessageBox.critical(self, "Fehler", message)

    # ------------------------------------------------------------------
    # OCR-Scanner
    # ------------------------------------------------------------------

    def _run_ocr_scan(self):
        text = self.input_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(
                self, "Kein Text", "Kein Text f\u00fcr den OCR-Scan vorhanden."
            )
            return

        self._ocr_findings_reviewed = False
        self._last_scan_result = self._ocr_scanner.scan(text)
        result = self._last_scan_result
        qualitaet = result["qualitaet"]
        gesamt = result["gesamt_fehler"]

        color_map = {
            "gut": "#006600",
            "mittel": "#cc6600",
            "schlecht": "#cc0000",
        }
        dot_map = {"gut": "\u25cf\u25cf\u25cf", "mittel": "\u25cf\u25cf\u25cb", "schlecht": "\u25cf\u25cb\u25cb"}
        color = color_map.get(qualitaet, "#333")
        dots = dot_map.get(qualitaet, "")

        self.ocr_quality_label.setText(
            f"Qualit\u00e4t: <span style='color:{color}; font-weight:bold;'>"
            f"{dots} {qualitaet}</span>"
        )
        self.ocr_quality_label.setVisible(True)

        if gesamt > 0:
            self.ocr_detail_btn.setText(f"| {gesamt} Fundstellen")
            self.ocr_detail_btn.setVisible(True)
        else:
            self.ocr_detail_btn.setVisible(False)

        self.status_bar.showMessage(
            f"OCR-Scan abgeschlossen \u2013 {gesamt} Fundstellen, Qualit\u00e4t: {qualitaet}"
        )

    def _show_ocr_fundstellen(self):
        if not self._last_scan_result:
            return
        dlg = OcrReviewDialog(
            scan_result=self._last_scan_result,
            main_text=self.input_text.toPlainText(),
            ollama_client=self.ollama_client,
            model=self.model_combo.currentText().strip(),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            corrected = dlg.get_corrected_text()
            if corrected != self.input_text.toPlainText():
                self.input_text.setPlainText(corrected)
                self.status_bar.showMessage("OCR-Korrekturen \u00fcbernommen.")
            self._ocr_findings_reviewed = True

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def _show_diff(self):
        original = self.input_text.toPlainText()
        modified = self.output_text.toPlainText()
        if not original and not modified:
            QMessageBox.information(
                self, "Diff", "Kein Text zum Vergleichen vorhanden."
            )
            return
        DiffWindow(original, modified, parent=self).exec()

    # ------------------------------------------------------------------
    # Dateioperationen
    # ------------------------------------------------------------------

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Datei \u00f6ffnen",
            "",
            "Unterst\u00fctzte Dateien (*.txt *.md *.epub);;"
            "Textdateien (*.txt *.md);;"
            "E-Books (*.epub);;"
            "Alle Dateien (*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            if path.lower().endswith(".epub"):
                content = EpubReader().read(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            self.input_text.setPlainText(content)
            self._input_file = path
            # Reset markers and workflow state for new file
            self._marker_start = None
            self._marker_end = None
            self.input_text.setExtraSelections([])
            self._output_up_to_date = False
            self._last_scan_result = None
            self._ocr_findings_reviewed = False
            self.settings.add_recent_file(path)
            self._update_recent_menu()
            self.status_bar.showMessage(f"Ge\u00f6ffnet: {path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Fehler", f"Datei konnte nicht ge\u00f6ffnet werden:\n{exc}"
            )

    def _save_output(self):
        text = self.output_text.toPlainText()
        if not text:
            QMessageBox.warning(self, "Kein Text", "Kein Ausgabetext zum Speichern.")
            return
        default = ""
        if self._input_file:
            p = Path(self._input_file)
            default = str(p.parent / f"{p.stem}_tts{p.suffix}")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Ausgabe speichern",
            default,
            "Textdateien (*.txt);;Markdown (*.md);;Alle Dateien (*)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.status_bar.showMessage(f"Gespeichert: {path}")
            except Exception as exc:
                QMessageBox.critical(
                    self, "Fehler", f"Datei konnte nicht gespeichert werden:\n{exc}"
                )

    def _update_recent_menu(self):
        self.recent_menu.clear()
        recent = self.settings.get_recent_files()
        if not recent:
            act = QAction("(keine)", self)
            act.setEnabled(False)
            self.recent_menu.addAction(act)
            return
        for path in recent:
            name = Path(path).name
            act = QAction(name, self)
            act.setToolTip(path)
            act.triggered.connect(lambda checked, p=path: self._open_recent_file(p))
            self.recent_menu.addAction(act)

    def _open_recent_file(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(
                self,
                "Datei nicht gefunden",
                f"Die Datei wurde nicht gefunden:\n{path}",
            )
            return
        self._load_file(path)

    # ------------------------------------------------------------------
    # TTS-Tab: Stimmprofile
    # ------------------------------------------------------------------

    def _refresh_voice_profiles(self):
        self._voice_combo.blockSignals(True)
        self._voice_combo.clear()
        for profile in self._voice_profile_manager.get_all():
            self._voice_combo.addItem(profile["name"], userData=profile["id"])
        self._voice_combo.blockSignals(False)

        if self._voice_combo.count() == 0:
            self._profile_form.setVisible(True)

    def _show_new_profile_form(self):
        self._profile_name_edit.clear()
        self._profile_wav_edit.clear()
        self._profile_txt_edit.clear()
        self._profile_form.setVisible(True)

    def _save_voice_profile(self):
        name = self._profile_name_edit.text().strip()
        if not name:
            QMessageBox.warning(
                self, "Kein Name", "Bitte geben Sie einen Profilnamen ein."
            )
            return
        wav = self._profile_wav_edit.text().strip()
        txt = self._profile_txt_edit.text().strip()
        self._voice_profile_manager.add(name, wav, txt)
        self._profile_form.setVisible(False)
        self._refresh_voice_profiles()
        idx = self._voice_combo.findText(name)
        if idx >= 0:
            self._voice_combo.setCurrentIndex(idx)

    def _delete_voice_profile(self):
        profile_id = self._voice_combo.currentData()
        if not profile_id:
            return
        name = self._voice_combo.currentText()
        reply = QMessageBox.question(
            self,
            "Profil l\u00f6schen",
            f'Stimmprofil \u201e{name}\u201c wirklich l\u00f6schen?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._voice_profile_manager.delete(profile_id)
            self._refresh_voice_profiles()

    def _browse_profile_wav(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Referenz-WAV ausw\u00e4hlen",
            "",
            "WAV-Dateien (*.wav);;Alle Dateien (*)",
        )
        if path:
            self._profile_wav_edit.setText(path)

    def _browse_profile_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Referenz-TXT ausw\u00e4hlen",
            "",
            "Textdateien (*.txt);;Alle Dateien (*)",
        )
        if path:
            self._profile_txt_edit.setText(path)

    # ------------------------------------------------------------------
    # TTS-Tab: Textquelle und Vorschau
    # ------------------------------------------------------------------

    def _tts_text_source_changed(self):
        self._update_tts_preview()

    def _update_tts_preview(self):
        if self._tts_radio_tab1.isChecked():
            text = self.output_text.toPlainText()
            if not text:
                text = self.input_text.toPlainText()
            self._tts_preview.setPlainText(text[:500])
        elif self._tts_input_file:
            try:
                with open(self._tts_input_file, encoding="utf-8") as f:
                    preview = f.read(500)
                self._tts_preview.setPlainText(preview)
            except Exception:
                self._tts_preview.setPlainText("")

    def _browse_tts_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Textdatei f\u00fcr TTS \u00f6ffnen",
            "",
            "Textdateien (*.txt *.md);;Alle Dateien (*)",
        )
        if path:
            self._tts_input_file = path
            self._tts_file_edit.setText(path)
            self._tts_radio_file.setChecked(True)
            self._update_tts_preview()

    def _browse_tts_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "TTS-Ausgabeordner w\u00e4hlen",
            self._tts_output_dir_edit.text(),
        )
        if path:
            self._tts_output_dir_edit.setText(path)

    # ------------------------------------------------------------------
    # TTS-Tab: TTS starten
    # ------------------------------------------------------------------

    def _start_tts(self):
        if self._check_processing_needed():
            return

        if self._has_open_ocr_findings():
            msg = QMessageBox(self)
            msg.setWindowTitle("Ungepr\u00fcfte OCR-Fundstellen")
            n = self._last_scan_result.get("gesamt_fehler", 0)  # type: ignore[union-attr]
            msg.setText(
                f"Es liegen {n}\u00a0ungepr\u00fcfte OCR-Fundstellen vor.\n\n"
                "M\u00f6chten Sie diese pr\u00fcfen, bevor Sie TTS starten?"
            )
            btn_review = msg.addButton(
                "Fundstellen pr\u00fcfen", QMessageBox.ButtonRole.AcceptRole
            )
            msg.addButton("Trotzdem fortfahren", QMessageBox.ButtonRole.DestructiveRole)
            msg.exec()
            if msg.clickedButton() == btn_review:
                self.tab_widget.setCurrentIndex(0)
                self._show_ocr_fundstellen()
                return

        if self._tts_radio_tab1.isChecked():
            text = self.output_text.toPlainText()
            if not text:
                text = self.input_text.toPlainText()
        else:
            if not self._tts_input_file:
                QMessageBox.warning(
                    self, "Keine Datei", "Bitte w\u00e4hlen Sie eine Textdatei aus."
                )
                return
            try:
                with open(self._tts_input_file, encoding="utf-8") as f:
                    text = f.read()
            except Exception as exc:
                QMessageBox.critical(
                    self, "Fehler", f"Datei konnte nicht gelesen werden:\n{exc}"
                )
                return

        if not text.strip():
            QMessageBox.warning(self, "Kein Text", "Kein Text f\u00fcr TTS vorhanden.")
            return

        output_dir = self._tts_output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "Kein Ausgabeordner",
                "Bitte w\u00e4hlen Sie einen Ausgabeordner."
            )
            return

        filename = self._tts_filename_edit.text().strip()
        if not filename:
            QMessageBox.warning(
                self, "Kein Dateiname", "Bitte geben Sie einen Dateinamen ein."
            )
            return

        formats: list[str] = []
        if self._fmt_mp3.isChecked():
            formats.append("mp3")
        if self._fmt_m4b.isChecked():
            formats.append("m4b")
        if self._fmt_wav.isChecked():
            formats.append("wav")
        if not formats:
            QMessageBox.warning(
                self, "Kein Format",
                "Bitte w\u00e4hlen Sie mindestens ein Ausgabeformat."
            )
            return

        self.settings.set("output_formats", formats)

        compose_file = self.settings.get(
            "compose_file",
            "F:\\german-audiobook-pipeline\\tts\\docker-compose.yml",
        )
        input_dir = str(Path(output_dir) / "_input")

        profile_id = self._voice_combo.currentData()
        voice_profile = None
        if profile_id:
            voice_profile = self._voice_profile_manager.get_by_id(profile_id)

        bridge = TtsBridge(
            compose_file=compose_file,
            input_dir=input_dir,
            output_dir=output_dir,
            voice_profile=voice_profile,
        )

        self._tts_start_btn.setEnabled(False)
        self._tts_progress_bar.setRange(0, 0)
        self._tts_progress_bar.setVisible(True)
        self._tts_progress_label.setText("TTS l\u00e4uft\u2026")
        self._tts_progress_label.setVisible(True)
        self.status_bar.showMessage("TTS-Verarbeitung gestartet\u2026")

        self._tts_worker = TtsWorker(bridge, text, filename, formats)
        self._tts_worker.progress.connect(self._on_tts_progress)
        self._tts_worker.finished.connect(self._on_tts_finished)
        self._tts_worker.error.connect(self._on_tts_error)
        self._tts_worker.start()

    def _on_tts_progress(self, msg: str):
        self._tts_progress_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_tts_finished(self, result: dict):
        self._tts_start_btn.setEnabled(True)
        self._tts_progress_bar.setRange(0, 100)
        self._tts_progress_bar.setValue(100)
        self._tts_progress_label.setText("Fertig!")
        self._tts_worker = None

        files = result.get("files", [])
        files_str = "\n".join(files) if files else "(keine Dateien)"
        self.status_bar.showMessage("TTS abgeschlossen.")
        QMessageBox.information(
            self,
            "TTS abgeschlossen",
            f"TTS-Verarbeitung erfolgreich abgeschlossen.\n\nAusgabedateien:\n{files_str}",
        )

    def _on_tts_error(self, msg: str):
        self._tts_start_btn.setEnabled(True)
        self._tts_progress_bar.setRange(0, 100)
        self._tts_progress_bar.setValue(0)
        self._tts_progress_bar.setVisible(False)
        self._tts_progress_label.setVisible(False)
        self._tts_worker = None
        self.status_bar.showMessage("TTS-Fehler.")
        QMessageBox.critical(self, "TTS-Fehler", msg)

    # ------------------------------------------------------------------
    # Einstellungen
    # ------------------------------------------------------------------

    def _open_settings(self):
        dlg = SettingsDialog(self.settings, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ollama_client = OllamaClient(
                self.settings.get("ollama_url", "http://localhost:11434")
            )
            self.model_combo.setCurrentText(
                self.settings.get("default_model", "gemma3:12b")
            )
            self.chunk_spin.setValue(self.settings.get("default_chunk_size", 2500))
            self._tts_output_dir_edit.setText(
                self.settings.get("tts_output_dir", "")
            )
            QTimer.singleShot(100, self._load_available_models)

    # ------------------------------------------------------------------
    # Sonderzeichen
    # ------------------------------------------------------------------

    def _toggle_invisibles(self, checked: bool) -> None:
        self._show_invisibles = checked

        apply_show_invisibles(self.input_text, checked)
        apply_show_invisibles(self.output_text, checked)

        if checked:
            self._hl_input = NonPrintableHighlighter(self.input_text.document())
            self._hl_output = NonPrintableHighlighter(self.output_text.document())
            self.invisibles_btn.setStyleSheet(
                "QPushButton { background:#555; color:#fff;"
                " border-radius:3px; padding:4px 10px; }"
                "QPushButton:hover { background:#333; }"
            )
        else:
            if self._hl_input:
                self._hl_input.setDocument(None)
                self._hl_input = None
            if self._hl_output:
                self._hl_output.setDocument(None)
                self._hl_output = None
            self.invisibles_btn.setStyleSheet("")

    # ------------------------------------------------------------------
    # Marker-Verwaltung
    # ------------------------------------------------------------------

    def _set_start_marker(self):
        pos = self.input_text.textCursor().position()
        self._marker_start = pos
        self._apply_marker_highlighting()
        self.status_bar.showMessage(f"Startmarker gesetzt (Position\u00a0{pos})")

    def _set_end_marker(self):
        pos = self.input_text.textCursor().position()
        self._marker_end = pos
        self._apply_marker_highlighting()
        self.status_bar.showMessage(f"Endmarker gesetzt (Position\u00a0{pos})")

    def _reset_markers(self):
        self._marker_start = None
        self._marker_end = None
        self.input_text.setExtraSelections([])
        self.status_bar.showMessage("Marker zur\u00fcckgesetzt.")

    def _apply_marker_highlighting(self):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#aaaaaa"))
        fmt.setFontItalic(True)

        full_len = len(self.input_text.toPlainText())
        selections: list[QTextEdit.ExtraSelection] = []

        def _make_sel(start: int, end: int) -> QTextEdit.ExtraSelection:
            sel = QTextEdit.ExtraSelection()
            cur = self.input_text.textCursor()
            cur.setPosition(start)
            cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            sel.cursor = cur
            sel.format = fmt
            return sel

        start = self._marker_start
        end = self._marker_end

        if start is not None and start > 0:
            selections.append(_make_sel(0, start))
        if end is not None and end < full_len:
            selections.append(_make_sel(end, full_len))

        self.input_text.setExtraSelections(selections)

    def _get_effective_text(self) -> str:
        """Gibt den Text zwischen den Markern zurück (oder den gesamten Text)."""
        text = self.input_text.toPlainText()
        if self._marker_start is None and self._marker_end is None:
            return text
        start = self._marker_start if self._marker_start is not None else 0
        end = self._marker_end if self._marker_end is not None else len(text)
        return text[start:end]

    # ------------------------------------------------------------------
    # Vorspann-Erkennung
    # ------------------------------------------------------------------

    def _detect_preamble(self):
        text = self.input_text.toPlainText()
        if not text.strip():
            QMessageBox.warning(
                self, "Kein Text",
                "Kein Text f\u00fcr die Vorspann-Erkennung vorhanden."
            )
            return
        model = self.model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(
                self, "Kein Modell", "Bitte w\u00e4hlen Sie ein Modell aus."
            )
            return

        self._btn_detect_preamble.setEnabled(False)
        self.status_bar.showMessage("Vorspann wird erkannt\u2026")

        self._vorspann_worker = VorspannWorker(self.ollama_client, model, text)
        self._vorspann_worker.detected.connect(self._on_vorspann_detected)
        self._vorspann_worker.error.connect(self._on_vorspann_error)
        self._vorspann_worker.start()

    def _on_vorspann_detected(self, first_content_line: str):
        self._btn_detect_preamble.setEnabled(True)
        self._vorspann_worker = None

        if not first_content_line:
            QMessageBox.information(
                self, "Kein Vorspann erkannt",
                "Das LLM hat keinen Vorspann im Text erkannt."
            )
            self.status_bar.showMessage("Kein Vorspann erkannt.")
            return

        text = self.input_text.toPlainText()
        pos = text.find(first_content_line)
        if pos == -1:
            QMessageBox.information(
                self, "Nicht gefunden",
                f"Der erkannte Inhaltsbeginn konnte im Text nicht gefunden werden:"
                f"\n\n\u201e{first_content_line}\u201c"
            )
            self.status_bar.showMessage("Vorspann-Marker konnte nicht gesetzt werden.")
            return

        self._marker_start = pos
        self._apply_marker_highlighting()
        self.status_bar.showMessage(
            f"Startmarker gesetzt: Vorspann erkannt (Position\u00a0{pos})"
        )

    def _on_vorspann_error(self, msg: str):
        self._btn_detect_preamble.setEnabled(True)
        self._vorspann_worker = None
        self.status_bar.showMessage("Fehler bei der Vorspann-Erkennung.")
        QMessageBox.critical(
            self, "Fehler",
            f"Vorspann-Erkennung fehlgeschlagen:\n{msg}"
        )

    # ------------------------------------------------------------------
    # TTS-Workflow-Prüfungen
    # ------------------------------------------------------------------

    def _has_open_ocr_findings(self) -> bool:
        """True wenn ein Scan-Ergebnis mit Fundstellen vorliegt und noch nicht geprüft wurde."""
        if not self._last_scan_result:
            return False
        if self._ocr_findings_reviewed:
            return False
        return self._last_scan_result.get("gesamt_fehler", 0) > 0

    def _check_processing_needed(self) -> bool:
        """
        Prüft ob der Eingabetext noch nicht verarbeitet wurde.
        Gibt True zurück wenn der Aufrufer abbrechen soll (Benutzer wird
        die Verarbeitung separat starten oder hat Abbrechen gewählt).
        """
        input_text = self._get_effective_text().strip()
        output_text = self.output_text.toPlainText().strip()

        if not input_text:
            return False
        if output_text and self._output_up_to_date:
            return False

        msg = QMessageBox(self)
        msg.setWindowTitle("Text noch nicht verarbeitet")
        msg.setText(
            "Der Eingabetext wurde noch nicht (vollst\u00e4ndig) verarbeitet.\n\n"
            "M\u00f6chten Sie den Text jetzt verarbeiten?"
        )
        btn_process = msg.addButton(
            "Jetzt verarbeiten", QMessageBox.ButtonRole.AcceptRole
        )
        msg.addButton("Trotzdem fortfahren", QMessageBox.ButtonRole.DestructiveRole)
        btn_cancel = msg.addButton("Abbrechen", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_process:
            self.tab_widget.setCurrentIndex(0)
            self._pending_tts_after_processing = True
            self._start_processing()
            return True
        if clicked == btn_cancel:
            self.tab_widget.setCurrentIndex(0)
            return True
        return False  # "Trotzdem fortfahren"

    # ------------------------------------------------------------------
    # Über / Schließen
    # ------------------------------------------------------------------

    def _show_about(self):
        QMessageBox.about(
            self,
            "\u00dcber TTS Studio",
            "TTS Studio\n\n"
            "Desktop-Anwendung zur Vorverarbeitung von Texten\n"
            "f\u00fcr deutschsprachige TTS-Systeme.\n\n"
            "Backend: Ollama (lokale KI-Inferenz)\n"
            "TTS: Qwen3-TTS via Docker\n"
            "UI: PyQt6",
        )

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.terminate()
            self._tts_worker.wait(2000)
        if self._vorspann_worker and self._vorspann_worker.isRunning():
            self._vorspann_worker.terminate()
            self._vorspann_worker.wait(2000)
        event.accept()
