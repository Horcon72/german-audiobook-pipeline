"""
Tests für ProcessingWorker._split_into_chunks und die Cache-Logik.
Der QThread-Teil wird ohne Qt-Eventloop über direkte Methodenaufrufe getestet.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ProcessingWorker importieren ohne Qt zu starten
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.main_window import ProcessingWorker


def _make_worker(**kwargs) -> ProcessingWorker:
    defaults = dict(
        text="Dummy",
        model="gemma3:12b",
        chunk_size=2500,
        ollama_client=MagicMock(),
        prompt_builder=MagicMock(),
    )
    defaults.update(kwargs)
    return ProcessingWorker(**defaults)


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks(unittest.TestCase):
    split = staticmethod(ProcessingWorker._split_into_chunks)

    def test_empty_text_returns_empty(self):
        self.assertEqual(self.split("", 100), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(self.split("   \n  ", 100), [])

    def test_single_paragraph_fits_in_one_chunk(self):
        result = self.split("Hallo Welt.", 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Hallo Welt.")

    def test_text_without_paragraphs_single_chunk(self):
        text = "a" * 50
        result = self.split(text, 100)
        self.assertEqual(len(result), 1)

    def test_two_paragraphs_fit_together(self):
        text = "Absatz eins.\n\nAbsatz zwei."
        result = self.split(text, 1000)
        self.assertEqual(len(result), 1)
        self.assertIn("Absatz eins.", result[0])
        self.assertIn("Absatz zwei.", result[0])

    def test_splits_when_size_exceeded(self):
        para = "X" * 60
        text = f"{para}\n\n{para}"
        result = self.split(text, 100)
        self.assertEqual(len(result), 2)

    def test_no_content_lost(self):
        paras = [f"Absatz {i} " + "x" * 40 for i in range(5)]
        text = "\n\n".join(paras)
        result = self.split(text, 100)
        joined = "\n\n".join(result)
        for p in paras:
            self.assertIn(p, joined)

    def test_chunk_separator_preserved(self):
        text = "Eins.\n\nZwei."
        result = self.split(text, 1000)
        self.assertIn("\n\n", result[0])

    def test_large_single_paragraph_stays_in_one_chunk(self):
        # A single paragraph larger than max_size cannot be split further
        big = "W" * 200
        result = self.split(big, 100)
        self.assertEqual(len(result), 1)

    def test_many_paragraphs_chunk_count(self):
        # 10 paragraphs of 100 chars each, max_size=250 → ~5 chunks
        paras = ["A" * 100] * 10
        text = "\n\n".join(paras)
        result = self.split(text, 250)
        self.assertGreater(len(result), 1)
        self.assertLessEqual(len(result), 10)

    def test_exact_boundary_no_split(self):
        # Two paragraphs whose combined size (with separator) equals max_size exactly
        p1 = "A" * 48
        p2 = "B" * 48
        # combined = 48 + 2 + 48 = 98; separator = "\n\n" = 2 chars
        result = self.split(f"{p1}\n\n{p2}", 98)
        self.assertEqual(len(result), 1)

    def test_one_over_boundary_splits(self):
        p1 = "A" * 48
        p2 = "B" * 49  # combined = 48 + 2 + 49 = 99 > 98
        result = self.split(f"{p1}\n\n{p2}", 98)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Cache-Logik
# ---------------------------------------------------------------------------

class TestProcessingWorkerCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _worker(self, input_file="test.txt"):
        return _make_worker(
            cache_dir=str(self.tmp),
            input_file=str(self.tmp / input_file),
        )

    def test_cache_file_path_returns_none_without_cache_dir(self):
        w = _make_worker()
        self.assertIsNone(w._cache_file_path())

    def test_cache_file_path_returns_none_without_input_file(self):
        w = _make_worker(cache_dir=str(self.tmp))
        self.assertIsNone(w._cache_file_path())

    def test_cache_file_path_correct_name(self):
        w = self._worker("meinbuch.txt")
        path = w._cache_file_path()
        self.assertIsNotNone(path)
        self.assertEqual(Path(path).name, "meinbuch_cache.json")

    def test_save_cache_creates_file(self):
        w = self._worker()
        w._save_cache(total=5, done=2, results=["r1", "r2"])
        cache_file = Path(w._cache_file_path())
        self.assertTrue(cache_file.exists())

    def test_save_cache_content_correct(self):
        w = self._worker()
        w._save_cache(total=5, done=3, results=["a", "b", "c"])
        data = json.loads(Path(w._cache_file_path()).read_text(encoding="utf-8"))
        self.assertEqual(data["gesamt_chunks"], 5)
        self.assertEqual(data["verarbeitete_chunks"], 3)
        self.assertEqual(data["ergebnisse"], ["a", "b", "c"])

    def test_save_cache_stores_input_file(self):
        w = self._worker("buch.txt")
        w._save_cache(total=1, done=1, results=["x"])
        data = json.loads(Path(w._cache_file_path()).read_text(encoding="utf-8"))
        self.assertIn("buch.txt", data["eingabe_datei"])

    def test_clear_cache_removes_file(self):
        w = self._worker()
        w._save_cache(total=1, done=1, results=["x"])
        w._clear_cache()
        self.assertFalse(Path(w._cache_file_path()).exists())

    def test_clear_cache_no_error_when_no_file(self):
        w = self._worker()
        # Should not raise even if file does not exist
        w._clear_cache()

    def test_save_cache_no_op_without_cache_dir(self):
        w = _make_worker()
        # Should not raise and should not create any file
        w._save_cache(total=1, done=1, results=["x"])


# ---------------------------------------------------------------------------
# Pause-Mechanismus
# ---------------------------------------------------------------------------

class TestProcessingWorkerPause(unittest.TestCase):
    def test_initially_not_paused(self):
        w = _make_worker()
        self.assertFalse(w.is_paused())

    def test_pause_sets_paused(self):
        w = _make_worker()
        w.pause()
        self.assertTrue(w.is_paused())

    def test_resume_clears_paused(self):
        w = _make_worker()
        w.pause()
        w.resume()
        self.assertFalse(w.is_paused())

    def test_cancel_unblocks_paused_worker(self):
        w = _make_worker()
        w.pause()
        w.cancel()
        # After cancel the event must be set so the wait() returns immediately
        self.assertTrue(w._paused.is_set())

    def test_cancel_sets_cancelled(self):
        w = _make_worker()
        w.cancel()
        self.assertTrue(w.is_cancelled())


if __name__ == "__main__":
    unittest.main()
