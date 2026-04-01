from .rules_manager import RulesManager

HEADER = """\
Du bist ein spezialisierter Textprozessor für deutschsprachige TTS-Systeme.
Gib NUR den korrigierten Text zurück. Keine Erklärungen, keine Kommentare,
keine Markdown-Formatierung. Exakt dieselbe Struktur wie der Input.\
"""

FOOTER = """\
Verändere niemals Gedankenstriche, Kommas oder korrekte Anführungszeichen.
Keine inhaltlichen Änderungen. Zeilenstruktur exakt beibehalten.\
"""


class PromptBuilder:
    def __init__(self, rules_manager: RulesManager):
        self.rules_manager = rules_manager

    def build_system_prompt(self) -> str:
        parts = [HEADER, ""]

        active_rules = sorted(
            self.rules_manager.get_active(),
            key=lambda r: r.get("prioritaet", 999)
        )

        for rule in active_rules:
            if rule.get("vorverarbeitung_code"):
                continue
            parts.append(f"## {rule['name']}")
            beschreibung = rule.get("beschreibung", "").strip()
            if beschreibung:
                parts.append(beschreibung)

            examples = rule.get("beispiele", [])
            if examples:
                parts.append("Beispiele:")
                for ex in examples:
                    inp = ex.get("input", "")
                    out = ex.get("output", "")
                    parts.append(f'  \u201e{inp}\u201c \u2192 \u201e{out}\u201c')

            anmerkungen = rule.get("anmerkungen", "").strip()
            if anmerkungen:
                parts.append(anmerkungen)

            parts.append("")

        parts.append(FOOTER)
        return "\n".join(parts)
