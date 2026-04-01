import uuid

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

KATEGORIEN = [
    "Satzzeichen",
    "Zahlen",
    "Abk\u00fcrzungen",
    "Datumsangaben",
    "Sonderzeichen",
    "Sonstiges",
]


class RuleEditorDialog(QDialog):
    def __init__(self, rule: dict = None, parent=None):
        super().__init__(parent)
        self._rule = rule
        self.result_rule: dict | None = None
        title = "Regel bearbeiten" if rule else "Neue Regel"
        self.setWindowTitle(title)
        self.setMinimumSize(620, 720)
        self._setup_ui()
        if rule:
            self._load_rule(rule)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # --- Basic fields ---
        form = QFormLayout()
        form.setSpacing(8)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Regelname (Pflichtfeld)")
        form.addRow("Name:", self.name_edit)

        self.kategorie_combo = QComboBox()
        self.kategorie_combo.addItems(KATEGORIEN)
        form.addRow("Kategorie:", self.kategorie_combo)

        self.prioritaet_spin = QSpinBox()
        self.prioritaet_spin.setRange(1, 9999)
        self.prioritaet_spin.setValue(10)
        self.prioritaet_spin.setToolTip(
            "Kleinere Zahl = h\u00f6here Priorit\u00e4t (wird zuerst im System-Prompt aufgef\u00fchrt)"
        )
        form.addRow("Priorit\u00e4t:", self.prioritaet_spin)

        self.aktiv_check = QCheckBox("Aktiv")
        self.aktiv_check.setChecked(True)
        form.addRow("", self.aktiv_check)

        layout.addLayout(form)

        # --- Description ---
        desc_label = QLabel("Beschreibung:")
        desc_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(desc_label)

        self.beschreibung_edit = QPlainTextEdit()
        self.beschreibung_edit.setPlaceholderText(
            "Regeltext — wird 1:1 in den System-Prompt \u00fcbernommen"
        )
        self.beschreibung_edit.setMaximumHeight(100)
        layout.addWidget(self.beschreibung_edit)

        # --- Example pairs ---
        ex_header_row = QHBoxLayout()
        ex_label = QLabel("Beispielpaare:")
        ex_label.setStyleSheet("font-weight: bold;")
        ex_header_row.addWidget(ex_label)
        ex_header_row.addStretch()
        add_row_btn = QPushButton("+ Zeile")
        add_row_btn.setFixedWidth(72)
        add_row_btn.clicked.connect(lambda: self._add_example_row())
        del_row_btn = QPushButton("\u2212 Zeile")
        del_row_btn.setFixedWidth(72)
        del_row_btn.clicked.connect(self._remove_example_row)
        ex_header_row.addWidget(add_row_btn)
        ex_header_row.addWidget(del_row_btn)
        layout.addLayout(ex_header_row)

        self.examples_table = QTableWidget(0, 2)
        self.examples_table.setHorizontalHeaderLabels(["Input", "Output"])
        self.examples_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.examples_table.setAlternatingRowColors(True)
        self.examples_table.setMinimumHeight(160)
        layout.addWidget(self.examples_table)

        # --- Notes ---
        notes_label = QLabel("Anmerkungen:")
        notes_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(notes_label)

        self.anmerkungen_edit = QPlainTextEdit()
        self.anmerkungen_edit.setPlaceholderText(
            "Sonderf\u00e4lle, Ausnahmen, Kasus-Logik \u2014 freitext"
        )
        self.anmerkungen_edit.setMaximumHeight(90)
        layout.addWidget(self.anmerkungen_edit)

        # --- Dialog buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Speichern")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Abbrechen")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_rule(self, rule: dict):
        self.name_edit.setText(rule.get("name", ""))

        kategorie = rule.get("kategorie", "Sonstiges")
        idx = self.kategorie_combo.findText(kategorie)
        if idx >= 0:
            self.kategorie_combo.setCurrentIndex(idx)

        self.prioritaet_spin.setValue(rule.get("prioritaet", 10))
        self.aktiv_check.setChecked(rule.get("aktiv", True))
        self.beschreibung_edit.setPlainText(rule.get("beschreibung", ""))
        self.anmerkungen_edit.setPlainText(rule.get("anmerkungen", ""))

        for ex in rule.get("beispiele", []):
            self._add_example_row(ex.get("input", ""), ex.get("output", ""))

    def _add_example_row(self, input_text: str = "", output_text: str = ""):
        row = self.examples_table.rowCount()
        self.examples_table.insertRow(row)
        self.examples_table.setItem(row, 0, QTableWidgetItem(input_text))
        self.examples_table.setItem(row, 1, QTableWidgetItem(output_text))
        self.examples_table.scrollToBottom()

    def _remove_example_row(self):
        selected = self.examples_table.currentRow()
        if selected >= 0:
            self.examples_table.removeRow(selected)
        elif self.examples_table.rowCount() > 0:
            self.examples_table.removeRow(self.examples_table.rowCount() - 1)

    def _save(self):
        name = self.name_edit.text().strip()
        if not name:
            self.name_edit.setStyleSheet("border: 2px solid #cc0000;")
            self.name_edit.setFocus()
            self.name_edit.setPlaceholderText("Pflichtfeld!")
            return
        self.name_edit.setStyleSheet("")

        examples = []
        for row in range(self.examples_table.rowCount()):
            inp_item = self.examples_table.item(row, 0)
            out_item = self.examples_table.item(row, 1)
            inp = inp_item.text().strip() if inp_item else ""
            out = out_item.text().strip() if out_item else ""
            if inp or out:
                examples.append({"input": inp, "output": out})

        self.result_rule = {
            "id": self._rule.get("id") if self._rule else str(uuid.uuid4()),
            "name": name,
            "kategorie": self.kategorie_combo.currentText(),
            "beschreibung": self.beschreibung_edit.toPlainText().strip(),
            "aktiv": self.aktiv_check.isChecked(),
            "prioritaet": self.prioritaet_spin.value(),
            "beispiele": examples,
            "anmerkungen": self.anmerkungen_edit.toPlainText().strip(),
        }
        self.accept()

    def get_rule(self) -> dict | None:
        return self.result_rule
