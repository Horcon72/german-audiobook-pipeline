import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from core.tts_bridge import TtsBridge, TtsBridgeError


def _bridge(**kwargs) -> TtsBridge:
    defaults = {
        "compose_file": "/app/docker-compose.yml",
        "input_dir": "/input",
        "output_dir": "/output",
    }
    defaults.update(kwargs)
    return TtsBridge(**defaults)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestTtsBridgeRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _bridge(self):
        return TtsBridge(
            compose_file=str(self.tmp / "docker-compose.yml"),
            input_dir=str(self.tmp / "input"),
            output_dir=str(self.tmp / "output"),
        )

    def _make_wav(self, name="output.wav"):
        out = self.tmp / "output"
        out.mkdir(parents=True, exist_ok=True)
        wav = out / name
        wav.write_bytes(b"RIFF")
        return wav

    @patch("core.tts_bridge.subprocess.run")
    def test_writes_input_txt_file(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self._make_wav()
        b = self._bridge()
        b.run("Hallo Welt", "test", ["wav"])
        input_file = self.tmp / "input" / "test.txt"
        self.assertTrue(input_file.exists())
        self.assertEqual(input_file.read_text(encoding="utf-8"), "Hallo Welt")

    @patch("core.tts_bridge.subprocess.run")
    def test_success_returns_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self._make_wav()
        result = self._bridge().run("Text", "out", ["wav"])
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])

    @patch("core.tts_bridge.subprocess.run")
    def test_docker_error_returns_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Docker error", stdout="")
        result = self._bridge().run("Text", "out", ["wav"])
        self.assertFalse(result["success"])
        self.assertIn("Docker", result["error"])

    @patch("core.tts_bridge.subprocess.run")
    def test_no_wav_output_returns_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # No WAV created in output dir
        (self.tmp / "output").mkdir(parents=True, exist_ok=True)
        result = self._bridge().run("Text", "out", ["wav"])
        self.assertFalse(result["success"])
        self.assertIn("WAV", result["error"])

    @patch("core.tts_bridge.subprocess.run")
    def test_wav_format_includes_wav_path(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        wav = self._make_wav("book.wav")
        result = self._bridge().run("Text", "book", ["wav"])
        self.assertIn(str(wav), result["files"])

    @patch("core.tts_bridge.subprocess.run")
    def test_timeout_returns_failure(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=3600)
        result = self._bridge().run("Text", "out", ["wav"])
        self.assertFalse(result["success"])
        self.assertIn("Zeitüberschreitung", result["error"])

    @patch("core.tts_bridge.subprocess.run")
    def test_docker_not_found_returns_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = self._bridge().run("Text", "out", ["wav"])
        self.assertFalse(result["success"])
        self.assertIn("Docker", result["error"])

    @patch("core.tts_bridge.subprocess.run")
    def test_progress_callback_called(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self._make_wav()
        messages = []
        self._bridge().run("Text", "out", ["wav"], progress_callback=messages.append)
        self.assertGreater(len(messages), 0)

    @patch("core.tts_bridge.subprocess.run")
    def test_docker_command_contains_compose_file(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        self._make_wav()
        b = self._bridge()
        b.run("Text", "out", ["wav"])
        cmd = mock_run.call_args[0][0]
        self.assertIn(str(self.tmp / "docker-compose.yml"), cmd)

    @patch("core.tts_bridge.subprocess.run")
    def test_mp3_format_calls_convert(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        self._make_wav("book.wav")
        b = self._bridge()
        with patch.object(b, "convert_to_mp3", return_value=True) as mock_mp3:
            b.run("Text", "book", ["mp3"])
        mock_mp3.assert_called_once()

    @patch("core.tts_bridge.subprocess.run")
    def test_m4b_format_calls_convert(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        self._make_wav("book.wav")
        b = self._bridge()
        with patch.object(b, "convert_to_m4b", return_value=True) as mock_m4b:
            b.run("Text", "book", ["m4b"])
        mock_m4b.assert_called_once()


# ---------------------------------------------------------------------------
# convert_to_mp3 / convert_to_m4b
# ---------------------------------------------------------------------------

class TestTtsBridgeConvert(unittest.TestCase):
    def setUp(self):
        self.b = _bridge()

    @patch("core.tts_bridge.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("core.tts_bridge.subprocess.run")
    def test_convert_mp3_success(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(self.b.convert_to_mp3("/in.wav", "/out.mp3"))

    @patch("core.tts_bridge.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("core.tts_bridge.subprocess.run")
    def test_convert_mp3_failure(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=1)
        self.assertFalse(self.b.convert_to_mp3("/in.wav", "/out.mp3"))

    @patch("core.tts_bridge.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("core.tts_bridge.subprocess.run")
    def test_convert_m4b_success(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(self.b.convert_to_m4b("/in.wav", "/out.m4b"))

    @patch("core.tts_bridge.shutil.which", return_value=None)
    def test_convert_mp3_no_ffmpeg_raises(self, _):
        with self.assertRaises(TtsBridgeError) as ctx:
            self.b.convert_to_mp3("/in.wav", "/out.mp3")
        self.assertIn("ffmpeg", str(ctx.exception).lower())

    @patch("core.tts_bridge.shutil.which", return_value=None)
    def test_convert_m4b_no_ffmpeg_raises(self, _):
        with self.assertRaises(TtsBridgeError):
            self.b.convert_to_m4b("/in.wav", "/out.m4b")

    @patch("core.tts_bridge.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("core.tts_bridge.subprocess.run")
    def test_mp3_ffmpeg_args(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        self.b.convert_to_mp3("/in.wav", "/out.mp3")
        cmd = mock_run.call_args[0][0]
        self.assertIn("ffmpeg", cmd[0])
        self.assertIn("/in.wav", cmd)
        self.assertIn("/out.mp3", cmd)

    @patch("core.tts_bridge.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("core.tts_bridge.subprocess.run")
    def test_m4b_uses_aac_codec(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        self.b.convert_to_m4b("/in.wav", "/out.m4b")
        cmd = mock_run.call_args[0][0]
        self.assertIn("aac", cmd)


if __name__ == "__main__":
    unittest.main()
