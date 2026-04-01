import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.rules_manager as rm_mod
from core.rules_manager import RulesManager


class _RulesBase(unittest.TestCase):
    """Hält den Patch über die gesamte Testmethode aktiv."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._rules_file = self.tmp / "rules.json"
        self._p1 = patch.object(rm_mod, "DATA_DIR", self.tmp)
        self._p2 = patch.object(rm_mod, "RULES_FILE", self._rules_file)
        self._p1.start()
        self._p2.start()

    def tearDown(self):
        self._p2.stop()
        self._p1.stop()
        self._tmp.cleanup()

    def _mgr(self):
        return RulesManager()

    def _empty_mgr(self):
        """Manager ohne Default-Regeln."""
        self._rules_file.write_text("[]", encoding="utf-8")
        return RulesManager()


class TestRulesManagerInit(_RulesBase):
    def test_default_rules_loaded_when_no_file(self):
        self.assertGreater(len(self._mgr().get_all()), 0)

    def test_rules_file_created_on_init(self):
        self._mgr()
        self.assertTrue(self._rules_file.exists())

    def test_all_default_rules_have_required_keys(self):
        for rule in self._mgr().get_all():
            for key in ("id", "name", "kategorie", "aktiv", "prioritaet"):
                self.assertIn(key, rule)

    def test_corrupt_file_falls_back_to_defaults(self):
        self._rules_file.write_text("not json", encoding="utf-8")
        self.assertGreater(len(self._mgr().get_all()), 0)


class TestRulesManagerCrud(_RulesBase):
    def setUp(self):
        super().setUp()
        self.m = self._empty_mgr()

    def _add(self, name="Testregel", prio=50, aktiv=True):
        return self.m.add({
            "name": name,
            "kategorie": "Sonstiges",
            "beschreibung": "Test",
            "aktiv": aktiv,
            "prioritaet": prio,
            "beispiele": [],
            "anmerkungen": "",
        })

    def test_add_returns_rule_with_id(self):
        rule = self._add()
        self.assertIn("id", rule)
        self.assertTrue(len(rule["id"]) > 0)

    def test_add_id_auto_generated(self):
        self.assertIsNotNone(self._add()["id"])

    def test_get_all_contains_added_rule(self):
        rule = self._add("MeineRegel")
        self.assertIn(rule["id"], [r["id"] for r in self.m.get_all()])

    def test_get_by_id_found(self):
        rule = self._add()
        found = self.m.get_by_id(rule["id"])
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "Testregel")

    def test_get_by_id_not_found(self):
        self.assertIsNone(self.m.get_by_id("nonexistent-id"))

    def test_get_by_id_returns_copy(self):
        rule = self._add()
        copy = self.m.get_by_id(rule["id"])
        copy["name"] = "Geändert"
        self.assertEqual(self.m.get_by_id(rule["id"])["name"], "Testregel")

    def test_update_persists(self):
        rule = self._add()
        rule["name"] = "Aktualisiert"
        self.assertTrue(self.m.update(rule))
        self.assertEqual(self.m.get_by_id(rule["id"])["name"], "Aktualisiert")

    def test_update_unknown_id_returns_false(self):
        self.assertFalse(self.m.update({"id": "ghost", "name": "X"}))

    def test_delete_removes_rule(self):
        rule = self._add()
        self.assertTrue(self.m.delete(rule["id"]))
        self.assertIsNone(self.m.get_by_id(rule["id"]))

    def test_delete_unknown_id_returns_false(self):
        self.assertFalse(self.m.delete("ghost-id"))

    def test_persistence_across_instances(self):
        rule = self._add("Persistent")
        m2 = self._mgr()
        self.assertIn(rule["id"], [r["id"] for r in m2.get_all()])


class TestRulesManagerActiveFilter(_RulesBase):
    def setUp(self):
        super().setUp()
        self.m = self._empty_mgr()

    def _add(self, name, aktiv):
        return self.m.add({"name": name, "kategorie": "X", "beschreibung": "",
                           "aktiv": aktiv, "prioritaet": 1,
                           "beispiele": [], "anmerkungen": ""})

    def test_get_active_excludes_inactive(self):
        self._add("Aktiv", True)
        self._add("Inaktiv", False)
        names = [r["name"] for r in self.m.get_active()]
        self.assertIn("Aktiv", names)
        self.assertNotIn("Inaktiv", names)

    def test_toggle_active_off(self):
        rule = self._add("R", True)
        self.m.toggle_active(rule["id"])
        self.assertFalse(self.m.get_by_id(rule["id"])["aktiv"])

    def test_toggle_active_on_again(self):
        rule = self._add("R", True)
        self.m.toggle_active(rule["id"])
        self.m.toggle_active(rule["id"])
        self.assertTrue(self.m.get_by_id(rule["id"])["aktiv"])


class TestRulesManagerPriority(_RulesBase):
    def setUp(self):
        super().setUp()
        self.m = self._empty_mgr()

    def _add(self, name, prio):
        return self.m.add({"name": name, "kategorie": "X", "beschreibung": "",
                           "aktiv": True, "prioritaet": prio,
                           "beispiele": [], "anmerkungen": ""})

    def test_move_up_swaps_priority(self):
        r1 = self._add("Erste", 1)
        r2 = self._add("Zweite", 2)
        self.m.move_up(r2["id"])
        self.assertLess(
            self.m.get_by_id(r2["id"])["prioritaet"],
            self.m.get_by_id(r1["id"])["prioritaet"],
        )

    def test_move_down_swaps_priority(self):
        r1 = self._add("Erste", 1)
        r2 = self._add("Zweite", 2)
        self.m.move_down(r1["id"])
        self.assertGreater(
            self.m.get_by_id(r1["id"])["prioritaet"],
            self.m.get_by_id(r2["id"])["prioritaet"],
        )

    def test_move_up_first_rule_returns_false(self):
        self.assertFalse(self.m.move_up(self._add("Erste", 1)["id"]))

    def test_move_down_last_rule_returns_false(self):
        self.assertFalse(self.m.move_down(self._add("Erste", 1)["id"]))

    def test_move_up_unknown_id_returns_false(self):
        self.assertFalse(self.m.move_up("ghost"))

    def test_move_down_unknown_id_returns_false(self):
        self.assertFalse(self.m.move_down("ghost"))


if __name__ == "__main__":
    unittest.main()
