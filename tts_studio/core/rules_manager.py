import json
import uuid
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
RULES_FILE = DATA_DIR / "rules.json"

DEFAULT_RULES = [
    {
        "id": str(uuid.uuid4()),
        "name": "Zeilen ohne Satzzeichen",
        "kategorie": "Satzzeichen",
        "beschreibung": (
            "Endet eine Zeile nicht mit .!?:,;… und folgt ein Zeilenumbruch, "
            "wird ein Punkt ans Ende gesetzt. Gilt für Titel, Kapitelüberschriften, Aufzählungen."
        ),
        "aktiv": True,
        "prioritaet": 1,
        "vorverarbeitung_code": True,
        "beispiele": [
            {
                "input": "Kapitel 2: Der 3-Uhr-Morgens-Marathon",
                "output": "Kapitel 2: Der 3-Uhr-Morgens-Marathon."
            },
            {
                "input": "Das Manifest der Schnurr-Diktatur: Warum deine Katze dich eigentlich nur duldet",
                "output": "Das Manifest der Schnurr-Diktatur: Warum deine Katze dich eigentlich nur duldet."
            }
        ],
        "anmerkungen": ""
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Kardinalzahlen und Prozent",
        "kategorie": "Zahlen",
        "beschreibung": (
            "Zahlen ausschreiben, Kasus grammatikalisch korrekt. "
            "'zu 1 %' → 'zu einem Prozent', '90 %' → 'neunzig Prozent'."
        ),
        "aktiv": True,
        "prioritaet": 2,
        "beispiele": [
            {"input": "zu 1 %", "output": "zu einem Prozent"},
            {"input": "90 %", "output": "neunzig Prozent"},
            {"input": "5 %", "output": "fünf Prozent"},
            {"input": "100 %", "output": "hundert Prozent"}
        ],
        "anmerkungen": "Grammatikalischen Kasus aus dem Kontext ableiten."
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Datumsangaben und Jahreszahlen",
        "kategorie": "Datumsangaben",
        "beschreibung": (
            "Kasus aus Kontext ableiten. 'am 12. Januar' → 'am zwölften Januar' (Dativ), "
            "'der 12. Januar' → 'der zwölfte Januar' (Nominativ). "
            "Jahreszahlen vor 2000 als Hunderterpaare: '1984' → 'neunzehnhundertvierundachtzig'. "
            "Ab 2000 als Tausender: '2025' → 'zweitausendfünfundzwanzig'."
        ),
        "aktiv": True,
        "prioritaet": 3,
        "beispiele": [
            {"input": "am 12. Januar", "output": "am zwölften Januar"},
            {"input": "der 12. Januar", "output": "der zwölfte Januar"},
            {"input": "den 5. März", "output": "den fünften März"},
            {"input": "des 3. April", "output": "des dritten April"},
            {"input": "1984", "output": "neunzehnhundertvierundachtzig"},
            {"input": "2025", "output": "zweitausendfünfundzwanzig"}
        ],
        "anmerkungen": (
            "Kasus-Signalwörter: am/beim/seit/zum → Dativ, den → Akkusativ, "
            "der/die/das → Nominativ, des → Genitiv."
        )
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Uhrzeiten",
        "kategorie": "Zahlen",
        "beschreibung": "Uhrzeiten ausschreiben: '03:17 Uhr' → 'drei Uhr siebzehn'.",
        "aktiv": True,
        "prioritaet": 4,
        "beispiele": [
            {"input": "03:17 Uhr", "output": "drei Uhr siebzehn"},
            {"input": "14:30 Uhr", "output": "vierzehn Uhr dreißig"},
            {"input": "00:00 Uhr", "output": "null Uhr null"},
            {"input": "12:00 Uhr", "output": "zwölf Uhr null"}
        ],
        "anmerkungen": ""
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Formeln",
        "kategorie": "Abkürzungen",
        "beschreibung": (
            "Mathematische und physikalische Formeln ausschreiben. "
            "'g≈9,81m/s²' → 'g ungefähr neun Komma acht eins Meter pro Sekunde zum Quadrat'."
        ),
        "aktiv": True,
        "prioritaet": 5,
        "beispiele": [
            {
                "input": "g≈9,81m/s²",
                "output": "g ungefähr neun Komma acht eins Meter pro Sekunde zum Quadrat"
            }
        ],
        "anmerkungen": ""
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Abkürzungen",
        "kategorie": "Abkürzungen",
        "beschreibung": (
            "Gebräuchliche Abkürzungen ausschreiben. "
            "Buchstabierte Akronyme mit Bindestrichen trennen (UN → U-N, USA → U-S-A, ADAC → A-D-A-C). "
            "Aussprechbare Akronyme als Wort belassen (NASA → Nasa, LASER → Laser). "
            "Kriterium: Wird im allgemeinen Sprachgebrauch als zusammenhängendes Wort gesprochen. "
            "Im Zweifel buchstabieren."
        ),
        "aktiv": True,
        "prioritaet": 6,
        "beispiele": [
            {"input": "z.B.", "output": "zum Beispiel"},
            {"input": "d.h.", "output": "das heißt"},
            {"input": "usw.", "output": "und so weiter"},
            {"input": "bzw.", "output": "beziehungsweise"},
            {"input": "ggf.", "output": "gegebenenfalls"},
            {"input": "ca.", "output": "circa"},
            {"input": "d.", "output": "dem"},
            {"input": "UN", "output": "U-N"},
            {"input": "USA", "output": "U-S-A"},
            {"input": "ADAC", "output": "A-D-A-C"},
            {"input": "NASA", "output": "Nasa"},
            {"input": "LASER", "output": "Laser"}
        ],
        "anmerkungen": (
            "d. nur im Datumskontext als 'dem' auflösen, z.B. 'am 3. d. M.' → 'am dritten dieses Monats'."
        )
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Anführungszeichen-Parität",
        "kategorie": "Sonderzeichen",
        "beschreibung": (
            "Prüfen ob öffnende und schließende Anführungszeichen paarweise vorhanden sind. "
            "Fehlendes Gegenstück korrigieren oder entfernen. "
            "Gedankenstriche, Kommas und korrekte Anführungszeichen niemals verändern."
        ),
        "aktiv": True,
        "prioritaet": 7,
        "beispiele": [
            {"input": "Er sagte: \u201eHallo.", "output": "Er sagte: \u201eHallo.\u201c"},
            {"input": 'Das ist "falsch', "output": 'Das ist "falsch"'}
        ],
        "anmerkungen": (
            "Nur fehlerhafte oder unpaarige Anf\u00fchrungszeichen korrigieren. "
            "Korrekte deutsche typografische Anf\u00fchrungszeichen \u201e und \u201c nicht ver\u00e4ndern."
        )
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Steuerzeichen und Sonderzeichen",
        "kategorie": "Sonderzeichen",
        "beschreibung": (
            "Emojis und nicht-druckbare Steuerzeichen (außer \\n, \\t) entfernen. "
            "Aufzählungszeichen (•, ▪, ◦) und Bindestrich - am Zeilenanfang als Aufzählungszeichen entfernen."
        ),
        "aktiv": True,
        "prioritaet": 8,
        "beispiele": [
            {"input": "• Erster Punkt", "output": "Erster Punkt"},
            {"input": "▪ Zweiter Punkt", "output": "Zweiter Punkt"},
            {"input": "- Dritter Punkt am Zeilenanfang", "output": "Dritter Punkt am Zeilenanfang"},
            {"input": "Toll! 😀", "output": "Toll!"}
        ],
        "anmerkungen": (
            "Nur Steuerzeichen und Sonderzeichen entfernen, die für TTS irrelevant oder störend sind. "
            "Zeilenumbrüche (\\n) und Tabulatoren (\\t) beibehalten."
        )
    }
]


class RulesManager:
    def __init__(self):
        self._rules: list[dict] = []
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not RULES_FILE.exists():
            self._rules = [dict(r) for r in DEFAULT_RULES]
            self._save()
        else:
            try:
                with open(RULES_FILE, "r", encoding="utf-8") as f:
                    self._rules = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._rules = [dict(r) for r in DEFAULT_RULES]
                self._save()

    def _save(self):
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(self._rules, f, ensure_ascii=False, indent=2)

    def get_all(self) -> list[dict]:
        return list(self._rules)

    def get_active(self) -> list[dict]:
        return [r for r in self._rules if r.get("aktiv", True)]

    def get_by_id(self, rule_id: str) -> Optional[dict]:
        for rule in self._rules:
            if rule["id"] == rule_id:
                return dict(rule)
        return None

    def add(self, rule: dict) -> dict:
        if not rule.get("id"):
            rule["id"] = str(uuid.uuid4())
        self._rules.append(rule)
        self._save()
        return rule

    def update(self, rule: dict) -> bool:
        for i, r in enumerate(self._rules):
            if r["id"] == rule["id"]:
                self._rules[i] = rule
                self._save()
                return True
        return False

    def delete(self, rule_id: str) -> bool:
        for i, r in enumerate(self._rules):
            if r["id"] == rule_id:
                self._rules.pop(i)
                self._save()
                return True
        return False

    def move_up(self, rule_id: str) -> bool:
        sorted_rules = sorted(self._rules, key=lambda r: r.get("prioritaet", 999))
        idx = next((i for i, r in enumerate(sorted_rules) if r["id"] == rule_id), None)
        if idx is None or idx == 0:
            return False
        r_curr = sorted_rules[idx]
        r_prev = sorted_rules[idx - 1]
        r_curr["prioritaet"], r_prev["prioritaet"] = r_prev["prioritaet"], r_curr["prioritaet"]
        self._update_rule_priorities(r_curr, r_prev)
        return True

    def move_down(self, rule_id: str) -> bool:
        sorted_rules = sorted(self._rules, key=lambda r: r.get("prioritaet", 999))
        idx = next((i for i, r in enumerate(sorted_rules) if r["id"] == rule_id), None)
        if idx is None or idx >= len(sorted_rules) - 1:
            return False
        r_curr = sorted_rules[idx]
        r_next = sorted_rules[idx + 1]
        r_curr["prioritaet"], r_next["prioritaet"] = r_next["prioritaet"], r_curr["prioritaet"]
        self._update_rule_priorities(r_curr, r_next)
        return True

    def _update_rule_priorities(self, *updated_rules: dict):
        updated_map = {r["id"]: r["prioritaet"] for r in updated_rules}
        for rule in self._rules:
            if rule["id"] in updated_map:
                rule["prioritaet"] = updated_map[rule["id"]]
        self._save()

    def toggle_active(self, rule_id: str) -> bool:
        for rule in self._rules:
            if rule["id"] == rule_id:
                rule["aktiv"] = not rule.get("aktiv", True)
                self._save()
                return True
        return False
