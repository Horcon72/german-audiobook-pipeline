import unittest
from unittest.mock import MagicMock

from core.prompt_builder import PromptBuilder, HEADER, FOOTER


def _make_builder(rules: list[dict]) -> PromptBuilder:
    mock_rm = MagicMock()
    mock_rm.get_active.return_value = rules
    return PromptBuilder(mock_rm)


def _rule(name="Testregel", prio=10, beschreibung="", beispiele=None,
          anmerkungen="", vorverarbeitung=False):
    r = {
        "id": "test-id",
        "name": name,
        "kategorie": "Sonstiges",
        "beschreibung": beschreibung,
        "aktiv": True,
        "prioritaet": prio,
        "beispiele": beispiele or [],
        "anmerkungen": anmerkungen,
    }
    if vorverarbeitung:
        r["vorverarbeitung_code"] = True
    return r


class TestPromptBuilderStructure(unittest.TestCase):
    def test_header_present(self):
        prompt = _make_builder([]).build_system_prompt()
        self.assertIn("TTS-Systeme", prompt)

    def test_footer_present(self):
        prompt = _make_builder([]).build_system_prompt()
        self.assertIn("Zeilenstruktur", prompt)

    def test_header_before_footer(self):
        prompt = _make_builder([]).build_system_prompt()
        self.assertLess(prompt.index(HEADER[:20]), prompt.index(FOOTER[:20]))

    def test_no_rules_contains_only_header_and_footer(self):
        prompt = _make_builder([]).build_system_prompt()
        # No ## headings expected
        self.assertNotIn("##", prompt)


class TestPromptBuilderRuleInclusion(unittest.TestCase):
    def test_rule_name_appears_as_heading(self):
        prompt = _make_builder([_rule("MeineRegel")]).build_system_prompt()
        self.assertIn("## MeineRegel", prompt)

    def test_rule_beschreibung_included(self):
        prompt = _make_builder([_rule(beschreibung="Zahlen ausschreiben")]).build_system_prompt()
        self.assertIn("Zahlen ausschreiben", prompt)

    def test_empty_beschreibung_not_added(self):
        prompt = _make_builder([_rule(beschreibung="")]).build_system_prompt()
        # Only heading, no extra blank content between heading and next section
        lines = prompt.splitlines()
        heading_idx = next(i for i, l in enumerate(lines) if "## Testregel" in l)
        # Next non-empty line should be FOOTER or another rule heading
        rest = [l for l in lines[heading_idx + 1:] if l.strip()]
        self.assertTrue(rest[0].startswith(FOOTER[:10]) or rest[0].startswith("##"))

    def test_rule_with_examples_included(self):
        rule = _rule(beispiele=[{"input": "1 %", "output": "ein Prozent"}])
        prompt = _make_builder([rule]).build_system_prompt()
        self.assertIn("Beispiele:", prompt)
        self.assertIn("1 %", prompt)
        self.assertIn("ein Prozent", prompt)

    def test_rule_with_anmerkungen_included(self):
        rule = _rule(anmerkungen="Nur im Datumskontext.")
        prompt = _make_builder([rule]).build_system_prompt()
        self.assertIn("Nur im Datumskontext.", prompt)

    def test_vorverarbeitung_rule_excluded(self):
        rule = _rule(name="NurCode", vorverarbeitung=True)
        prompt = _make_builder([rule]).build_system_prompt()
        self.assertNotIn("## NurCode", prompt)

    def test_multiple_rules_all_present(self):
        rules = [_rule(f"Regel{i}", prio=i) for i in range(1, 4)]
        prompt = _make_builder(rules).build_system_prompt()
        for i in range(1, 4):
            self.assertIn(f"## Regel{i}", prompt)


class TestPromptBuilderSorting(unittest.TestCase):
    def test_rules_sorted_by_priority_ascending(self):
        rules = [
            _rule("Zweite", prio=2),
            _rule("Erste", prio=1),
            _rule("Dritte", prio=3),
        ]
        prompt = _make_builder(rules).build_system_prompt()
        idx1 = prompt.index("## Erste")
        idx2 = prompt.index("## Zweite")
        idx3 = prompt.index("## Dritte")
        self.assertLess(idx1, idx2)
        self.assertLess(idx2, idx3)

    def test_missing_priority_treated_as_999(self):
        rule_no_prio = {"id": "x", "name": "OhnePrio", "kategorie": "X",
                        "beschreibung": "", "aktiv": True, "beispiele": [],
                        "anmerkungen": ""}
        rule_with_prio = _rule("MitPrio", prio=500)
        prompt = _make_builder([rule_no_prio, rule_with_prio]).build_system_prompt()
        self.assertLess(prompt.index("## MitPrio"), prompt.index("## OhnePrio"))


if __name__ == "__main__":
    unittest.main()
