import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.settings_manager as sm_mod
from core.settings_manager import SettingsManager


class _SettingsBase(unittest.TestCase):
    """Hält den Patch über die gesamte Testmethode aktiv."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._settings_file = self.tmp / "settings.json"
        self._p1 = patch.object(sm_mod, "DATA_DIR", self.tmp)
        self._p2 = patch.object(sm_mod, "SETTINGS_FILE", self._settings_file)
        self._p1.start()
        self._p2.start()

    def tearDown(self):
        self._p2.stop()
        self._p1.stop()
        self._tmp.cleanup()

    def _mgr(self):
        return SettingsManager()


class TestSettingsManagerDefaults(_SettingsBase):
    def test_default_ollama_url(self):
        self.assertEqual(self._mgr().get("ollama_url"), "http://localhost:11434")

    def test_default_model(self):
        self.assertEqual(self._mgr().get("default_model"), "gemma3:12b")

    def test_default_chunk_size(self):
        self.assertEqual(self._mgr().get("default_chunk_size"), 2500)

    def test_default_cache_dir_empty(self):
        self.assertEqual(self._mgr().get("cache_dir"), "")

    def test_default_output_formats(self):
        self.assertEqual(self._mgr().get("output_formats"), ["mp3"])

    def test_default_tts_output_dir_empty(self):
        self.assertEqual(self._mgr().get("tts_output_dir"), "")

    def test_default_recent_files_empty(self):
        self.assertEqual(self._mgr().get("recent_files"), [])

    def test_missing_key_returns_none(self):
        self.assertIsNone(self._mgr().get("nonexistent"))

    def test_missing_key_with_fallback(self):
        self.assertEqual(self._mgr().get("nonexistent", "fallback"), "fallback")


class TestSettingsManagerPersistence(_SettingsBase):
    def test_set_persists_across_instances(self):
        m1 = self._mgr()
        m1.set("default_model", "llama3:8b")
        m2 = self._mgr()
        self.assertEqual(m2.get("default_model"), "llama3:8b")

    def test_set_multiple_keys(self):
        m = self._mgr()
        m.set("cache_dir", "/tmp/cache")
        m.set("tts_output_dir", "/tmp/out")
        m2 = self._mgr()
        self.assertEqual(m2.get("cache_dir"), "/tmp/cache")
        self.assertEqual(m2.get("tts_output_dir"), "/tmp/out")

    def test_settings_file_is_valid_json(self):
        m = self._mgr()
        m.set("default_model", "test")
        data = json.loads(self._settings_file.read_text(encoding="utf-8"))
        self.assertEqual(data["default_model"], "test")

    def test_corrupt_file_falls_back_to_defaults(self):
        self._settings_file.write_text("{ invalid json }", encoding="utf-8")
        m = self._mgr()
        self.assertEqual(m.get("ollama_url"), "http://localhost:11434")


class TestSettingsManagerRecentFiles(_SettingsBase):
    def test_add_recent_file(self):
        m = self._mgr()
        m.add_recent_file("/path/to/file.txt")
        self.assertEqual(m.get_recent_files()[0], "/path/to/file.txt")

    def test_recent_file_at_front(self):
        m = self._mgr()
        m.add_recent_file("/first.txt")
        m.add_recent_file("/second.txt")
        self.assertEqual(m.get_recent_files()[0], "/second.txt")

    def test_recent_files_max_5(self):
        m = self._mgr()
        for i in range(8):
            m.add_recent_file(f"/file{i}.txt")
        self.assertEqual(len(m.get_recent_files()), 5)

    def test_recent_files_deduplication(self):
        m = self._mgr()
        m.add_recent_file("/file.txt")
        m.add_recent_file("/other.txt")
        m.add_recent_file("/file.txt")
        recent = m.get_recent_files()
        self.assertEqual(recent[0], "/file.txt")
        self.assertEqual(recent.count("/file.txt"), 1)

    def test_get_recent_files_returns_copy(self):
        m = self._mgr()
        m.add_recent_file("/file.txt")
        copy = m.get_recent_files()
        copy.clear()
        self.assertEqual(len(m.get_recent_files()), 1)


if __name__ == "__main__":
    unittest.main()
