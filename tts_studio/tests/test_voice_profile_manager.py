import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.voice_profile_manager as vpm_mod
from core.voice_profile_manager import VoiceProfileManager


class _ProfileBase(unittest.TestCase):
    """Hält den Patch über die gesamte Testmethode aktiv."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._profiles_file = self.tmp / "voice_profiles.json"
        self._p1 = patch.object(vpm_mod, "DATA_DIR", self.tmp)
        self._p2 = patch.object(vpm_mod, "PROFILES_FILE", self._profiles_file)
        self._p1.start()
        self._p2.start()

    def tearDown(self):
        self._p2.stop()
        self._p1.stop()
        self._tmp.cleanup()

    def _mgr(self):
        return VoiceProfileManager()


class TestVoiceProfileManagerAdd(_ProfileBase):
    def setUp(self):
        super().setUp()
        self.m = self._mgr()

    def test_add_returns_dict(self):
        self.assertIsInstance(self.m.add("Stimme A", "/wav/a.wav", "/txt/a.txt"), dict)

    def test_add_assigns_uuid(self):
        p = self.m.add("Stimme A", "/wav/a.wav", "/txt/a.txt")
        self.assertEqual(len(p["id"]), 36)

    def test_add_stores_name(self):
        p = self.m.add("Stimme B", "", "")
        self.assertEqual(p["name"], "Stimme B")

    def test_add_stores_paths(self):
        p = self.m.add("X", "/my/ref.wav", "/my/ref.txt")
        self.assertEqual(p["wav_path"], "/my/ref.wav")
        self.assertEqual(p["txt_path"], "/my/ref.txt")

    def test_add_stores_timestamp(self):
        p = self.m.add("X", "", "")
        self.assertTrue(len(p["erstellt_am"]) > 0)

    def test_add_increments_count(self):
        self.m.add("A", "", "")
        self.m.add("B", "", "")
        self.assertEqual(len(self.m.get_all()), 2)


class TestVoiceProfileManagerGet(_ProfileBase):
    def setUp(self):
        super().setUp()
        self.m = self._mgr()
        self.profile = self.m.add("Stimme C", "/wav/c.wav", "/txt/c.txt")

    def test_get_all_contains_added(self):
        self.assertIn(self.profile["id"], [p["id"] for p in self.m.get_all()])

    def test_get_all_returns_copy(self):
        copy = self.m.get_all()
        copy.clear()
        self.assertEqual(len(self.m.get_all()), 1)

    def test_get_by_id_found(self):
        found = self.m.get_by_id(self.profile["id"])
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "Stimme C")

    def test_get_by_id_not_found(self):
        self.assertIsNone(self.m.get_by_id("no-such-id"))

    def test_get_by_id_returns_copy(self):
        copy = self.m.get_by_id(self.profile["id"])
        copy["name"] = "Geändert"
        self.assertEqual(self.m.get_by_id(self.profile["id"])["name"], "Stimme C")


class TestVoiceProfileManagerDelete(_ProfileBase):
    def setUp(self):
        super().setUp()
        self.m = self._mgr()

    def test_delete_removes_profile(self):
        p = self.m.add("Löschen", "", "")
        self.assertTrue(self.m.delete(p["id"]))
        self.assertIsNone(self.m.get_by_id(p["id"]))

    def test_delete_returns_false_for_unknown_id(self):
        self.assertFalse(self.m.delete("ghost-id"))

    def test_delete_only_removes_target(self):
        p1 = self.m.add("Bleibe", "", "")
        p2 = self.m.add("Weg", "", "")
        self.m.delete(p2["id"])
        self.assertIsNotNone(self.m.get_by_id(p1["id"]))


class TestVoiceProfileManagerPersistence(_ProfileBase):
    def test_profiles_persist_across_instances(self):
        m1 = self._mgr()
        p = m1.add("Persistent", "/wav/p.wav", "/txt/p.txt")
        m2 = self._mgr()
        self.assertIn(p["id"], [x["id"] for x in m2.get_all()])

    def test_delete_persists_across_instances(self):
        m1 = self._mgr()
        p = m1.add("Weg", "", "")
        m1.delete(p["id"])
        self.assertIsNone(self._mgr().get_by_id(p["id"]))

    def test_profiles_file_is_valid_json(self):
        m = self._mgr()
        m.add("Test", "/wav/t.wav", "/txt/t.txt")
        data = json.loads(self._profiles_file.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["name"], "Test")

    def test_empty_file_starts_without_error(self):
        self._profiles_file.write_text("[]", encoding="utf-8")
        self.assertEqual(self._mgr().get_all(), [])

    def test_corrupt_file_starts_empty(self):
        self._profiles_file.write_text("not json", encoding="utf-8")
        self.assertEqual(self._mgr().get_all(), [])


if __name__ == "__main__":
    unittest.main()
