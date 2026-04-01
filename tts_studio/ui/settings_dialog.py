from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.settings = settings_manager
        self.setWindowTitle("Einstellungen")
        self.setMinimumWidth(520)
        self._setup_ui()
        self._load_values()

    # ------------------------------------------------------------------

    def _make_path_row(self, placeholder: str) -> tuple[QLineEdit, QHBoxLayout]:
        """Erstellt ein Pfad-Eingabefeld mit Durchsuchen-Button."""
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(edit, stretch=1)
        return edit, row

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        # Ollama
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("http://localhost:11434")
        form.addRow("Ollama-URL:", self.url_edit)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("gemma3:12b")
        form.addRow("Standard-Modell:", self.model_edit)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(500, 10000)
        self.chunk_spin.setSingleStep(500)
        self.chunk_spin.setSuffix(" Zeichen")
        form.addRow("Standard-Chunk-Größe:", self.chunk_spin)

        # Zwischenspeicher-Ordner
        self.cache_dir_edit, cache_row = self._make_path_row(
            "Ordner für Fortschritts-Zwischenspeicher"
        )
        cache_browse = QPushButton("Durchsuchen…")
        cache_browse.setFixedWidth(100)
        cache_browse.clicked.connect(self._browse_cache_dir)
        cache_row.addWidget(cache_browse)
        form.addRow("Zwischenspeicher-Ordner:", cache_row)

        # TTS-Ausgabe-Ordner
        self.tts_output_dir_edit, tts_row = self._make_path_row(
            "Standard-Ausgabeordner für TTS-Dateien"
        )
        tts_browse = QPushButton("Durchsuchen…")
        tts_browse.setFixedWidth(100)
        tts_browse.clicked.connect(self._browse_tts_output_dir)
        tts_row.addWidget(tts_browse)
        form.addRow("TTS-Ausgabe-Ordner:", tts_row)

        # Docker Compose Pfad
        self.compose_file_edit, compose_row = self._make_path_row(
            "Pfad zur docker-compose.yml"
        )
        compose_browse = QPushButton("Durchsuchen…")
        compose_browse.setFixedWidth(100)
        compose_browse.clicked.connect(self._browse_compose_file)
        compose_row.addWidget(compose_browse)
        form.addRow("Docker Compose Pfad:", compose_row)

        layout.addLayout(form)

        hint = QLabel(
            "<small style='color:#777;'>"
            "Änderungen werden sofort wirksam."
            "</small>"
        )
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------

    def _load_values(self):
        self.url_edit.setText(
            self.settings.get("ollama_url", "http://localhost:11434")
        )
        self.model_edit.setText(self.settings.get("default_model", "gemma3:12b"))
        self.chunk_spin.setValue(self.settings.get("default_chunk_size", 2500))
        self.cache_dir_edit.setText(self.settings.get("cache_dir", ""))
        self.tts_output_dir_edit.setText(self.settings.get("tts_output_dir", ""))
        self.compose_file_edit.setText(self.settings.get("compose_file", ""))

    def _save(self):
        url = self.url_edit.text().strip() or "http://localhost:11434"
        model = self.model_edit.text().strip() or "gemma3:12b"
        self.settings.set("ollama_url", url)
        self.settings.set("default_model", model)
        self.settings.set("default_chunk_size", self.chunk_spin.value())
        self.settings.set("cache_dir", self.cache_dir_edit.text().strip())
        self.settings.set("tts_output_dir", self.tts_output_dir_edit.text().strip())
        self.settings.set("compose_file", self.compose_file_edit.text().strip())
        self.accept()

    # ------------------------------------------------------------------
    # Browse-Dialoge
    # ------------------------------------------------------------------

    def _browse_cache_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Zwischenspeicher-Ordner wählen", self.cache_dir_edit.text()
        )
        if path:
            self.cache_dir_edit.setText(path)

    def _browse_tts_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "TTS-Ausgabe-Ordner wählen", self.tts_output_dir_edit.text()
        )
        if path:
            self.tts_output_dir_edit.setText(path)

    def _browse_compose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "docker-compose.yml auswählen",
            self.compose_file_edit.text(),
            "YAML-Dateien (*.yml *.yaml);;Alle Dateien (*)",
        )
        if path:
            self.compose_file_edit.setText(path)
