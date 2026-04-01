import unittest
from unittest.mock import MagicMock, patch

from core.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    preprocess_text,
)


# ---------------------------------------------------------------------------
# preprocess_text
# ---------------------------------------------------------------------------

class TestPreprocessText(unittest.TestCase):
    def test_adds_period_to_bare_line(self):
        self.assertEqual(preprocess_text("Hallo"), "Hallo.")

    def test_no_period_if_ends_with_period(self):
        self.assertEqual(preprocess_text("Hallo."), "Hallo.")

    def test_no_period_if_ends_with_exclamation(self):
        self.assertEqual(preprocess_text("Hallo!"), "Hallo!")

    def test_no_period_if_ends_with_question(self):
        self.assertEqual(preprocess_text("Hallo?"), "Hallo?")

    def test_no_period_if_ends_with_comma(self):
        self.assertEqual(preprocess_text("Hallo,"), "Hallo,")

    def test_no_period_if_ends_with_ellipsis(self):
        self.assertEqual(preprocess_text("Hallo\u2026"), "Hallo\u2026")

    def test_no_period_if_ends_with_opening_guillemet(self):
        # Only » (U+00BB) is in SATZZEICHEN; a line ending with » is not extended
        self.assertEqual(preprocess_text('\u00bbHallo\u00bb'), '\u00bbHallo\u00bb')

    def test_multiline_bare_and_punctuated(self):
        result = preprocess_text("Zeile eins\nZeile zwei.")
        self.assertEqual(result, "Zeile eins.\nZeile zwei.")

    def test_empty_line_not_modified(self):
        result = preprocess_text("Zeile\n\nNach Leerzeile")
        lines = result.splitlines()
        self.assertEqual(lines[1], "")

    def test_whitespace_only_line_not_modified(self):
        result = preprocess_text("   ")
        self.assertEqual(result, "   ")

    def test_trailing_whitespace_stripped_before_check(self):
        result = preprocess_text("Hallo   ")
        self.assertEqual(result.strip(), "Hallo.")


# ---------------------------------------------------------------------------
# OllamaClient.generate()
# ---------------------------------------------------------------------------

def _mock_response(lines: list[bytes], status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status = MagicMock()
    return resp


class TestOllamaClientGenerate(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient("http://localhost:11434")

    @patch("core.ollama_client.requests.post")
    def test_returns_combined_response(self, mock_post):
        mock_post.return_value = _mock_response([
            b'{"response": "Hallo", "done": false}',
            b'{"response": " Welt", "done": true}',
        ])
        result = self.client.generate("gemma3:12b", "system", "user")
        self.assertEqual(result, "Hallo Welt")

    @patch("core.ollama_client.requests.post")
    def test_stops_at_done_true(self, mock_post):
        mock_post.return_value = _mock_response([
            b'{"response": "Teil1", "done": true}',
            b'{"response": "Teil2", "done": false}',  # should never be read
        ])
        result = self.client.generate("gemma3:12b", "system", "user")
        self.assertEqual(result, "Teil1")

    @patch("core.ollama_client.requests.post")
    def test_skips_empty_lines(self, mock_post):
        mock_post.return_value = _mock_response([
            b"",
            b'{"response": "OK", "done": true}',
        ])
        result = self.client.generate("gemma3:12b", "system", "user")
        self.assertEqual(result, "OK")

    @patch("core.ollama_client.requests.post")
    def test_skips_invalid_json_lines(self, mock_post):
        mock_post.return_value = _mock_response([
            b"not json at all",
            b'{"response": "OK", "done": true}',
        ])
        result = self.client.generate("gemma3:12b", "system", "user")
        self.assertEqual(result, "OK")

    @patch("core.ollama_client.requests.post")
    def test_connection_error_raises(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with self.assertRaises(OllamaConnectionError):
            self.client.generate("gemma3:12b", "system", "user")

    @patch("core.ollama_client.requests.post")
    def test_404_raises_model_not_found(self, mock_post):
        mock_post.return_value = _mock_response([], status_code=404)
        with patch.object(self.client, "list_models", return_value=["model-a"]):
            with self.assertRaises(OllamaModelNotFoundError) as ctx:
                self.client.generate("ghost-model", "system", "user")
        self.assertEqual(ctx.exception.model, "ghost-model")
        self.assertIn("model-a", ctx.exception.available)

    @patch("core.ollama_client.requests.post")
    def test_cancelled_callback_stops_stream(self, mock_post):
        # Returns empty immediately when cancelled
        mock_post.return_value = _mock_response([
            b'{"response": "Teil", "done": false}',
        ])
        result = self.client.generate(
            "gemma3:12b", "system", "user",
            cancelled_callback=lambda: True,
        )
        self.assertEqual(result, "")

    @patch("core.ollama_client.requests.post")
    def test_payload_contains_model_and_prompt(self, mock_post):
        mock_post.return_value = _mock_response([
            b'{"response": "x", "done": true}',
        ])
        self.client.generate("mymodel", "sys", "usr")
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "mymodel")
        self.assertEqual(payload["system"], "sys")
        self.assertEqual(payload["prompt"], "usr")
        self.assertTrue(payload["stream"])


# ---------------------------------------------------------------------------
# Qwen3 think-Flag
# ---------------------------------------------------------------------------

class TestQwen3ThinkFlag(unittest.TestCase):
    @patch("core.ollama_client.requests.post")
    def _call(self, model_name, mock_post):
        mock_post.return_value = _mock_response([
            b'{"response": "x", "done": true}',
        ])
        OllamaClient().generate(model_name, "sys", "usr")
        _, kwargs = mock_post.call_args
        return kwargs["json"]

    def test_qwen3_lowercase_adds_think_false(self):
        payload = self._call("qwen3:7b")
        self.assertIn("options", payload)
        self.assertFalse(payload["options"]["think"])

    def test_qwen3_uppercase_adds_think_false(self):
        payload = self._call("Qwen3:14b")
        self.assertFalse(payload["options"]["think"])

    def test_qwen3_mixed_case_adds_think_false(self):
        payload = self._call("QWEN3-turbo")
        self.assertFalse(payload["options"]["think"])

    def test_non_qwen3_model_no_options(self):
        payload = self._call("gemma3:12b")
        self.assertNotIn("options", payload)

    def test_qwen2_model_no_options(self):
        payload = self._call("qwen2:7b")
        self.assertNotIn("options", payload)


# ---------------------------------------------------------------------------
# OllamaClient.list_models()
# ---------------------------------------------------------------------------

class TestOllamaClientListModels(unittest.TestCase):
    @patch("core.ollama_client.requests.get")
    def test_returns_model_names(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "gemma3:12b"}, {"name": "qwen3:7b"}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        result = OllamaClient().list_models()
        self.assertEqual(result, ["gemma3:12b", "qwen3:7b"])

    @patch("core.ollama_client.requests.get")
    def test_returns_empty_on_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        result = OllamaClient().list_models()
        self.assertEqual(result, [])

    @patch("core.ollama_client.requests.get")
    def test_empty_models_list(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        self.assertEqual(OllamaClient().list_models(), [])


if __name__ == "__main__":
    unittest.main()
