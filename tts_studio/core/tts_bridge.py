import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


class TtsBridgeError(Exception):
    pass


class TtsBridge:
    """Übergabe von Texten an den Qwen3-TTS Docker-Container."""

    def __init__(
        self,
        compose_file: str,
        input_dir: str,
        output_dir: str,
        voice_profile: Optional[dict] = None,
    ):
        self.compose_file = compose_file
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.voice_profile = voice_profile

    def run(
        self,
        text: str,
        filename: str,
        output_formats: list[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Speichert text als .txt in input_dir, ruft Docker auf,
        konvertiert WAV-Output in gewählte Formate (mp3, m4b, wav).
        Gibt zurück: {"success": bool, "files": [...], "error": str | None}
        """
        try:
            input_path = Path(self.input_dir) / f"{filename}.txt"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(text)

            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback("Starte Docker-Container…")

            cmd = [
                "docker", "compose",
                "-f", self.compose_file,
                "run", "--rm",
                "-v", f"{self.input_dir}:/input",
                "-v", f"{self.output_dir}:/output",
                "qwen-tts",
                "python", "audiobook.py",
                f"/input/{filename}.txt",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "files": [],
                    "error": f"Docker-Fehler:\n{result.stderr or result.stdout}",
                }

            if progress_callback:
                progress_callback("Docker abgeschlossen, konvertiere Audio…")

            wav_files = list(Path(self.output_dir).glob("*.wav"))
            if not wav_files:
                return {
                    "success": False,
                    "files": [],
                    "error": "Kein WAV-Output im Ausgabeordner gefunden.",
                }

            output_files: list[str] = []
            for wav_path in wav_files:
                stem = wav_path.stem
                for fmt in output_formats:
                    fmt_lower = fmt.lower()
                    if fmt_lower == "wav":
                        output_files.append(str(wav_path))
                    elif fmt_lower == "mp3":
                        mp3_path = wav_path.parent / f"{stem}.mp3"
                        if self.convert_to_mp3(str(wav_path), str(mp3_path)):
                            output_files.append(str(mp3_path))
                    elif fmt_lower == "m4b":
                        m4b_path = wav_path.parent / f"{stem}.m4b"
                        if self.convert_to_m4b(str(wav_path), str(m4b_path)):
                            output_files.append(str(m4b_path))

            return {"success": True, "files": output_files, "error": None}

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "files": [],
                "error": "Zeitüberschreitung beim Docker-Aufruf (> 1 Stunde).",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "files": [],
                "error": (
                    "Docker nicht gefunden.\n"
                    "Bitte Docker Desktop installieren und sicherstellen, "
                    "dass 'docker' im PATH verfügbar ist."
                ),
            }
        except Exception as exc:
            return {"success": False, "files": [], "error": str(exc)}

    # ------------------------------------------------------------------

    @staticmethod
    def _check_ffmpeg() -> bool:
        return shutil.which("ffmpeg") is not None

    def convert_to_mp3(self, wav_path: str, output_path: str) -> bool:
        """Konvertiert WAV zu MP3 via ffmpeg."""
        if not self._check_ffmpeg():
            raise TtsBridgeError(
                "ffmpeg nicht gefunden.\n"
                "Bitte ffmpeg installieren und im PATH verfügbar machen.\n"
                "Download: https://ffmpeg.org/download.html"
            )
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", output_path],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def convert_to_m4b(self, wav_path: str, output_path: str) -> bool:
        """Konvertiert WAV zu M4B via ffmpeg."""
        if not self._check_ffmpeg():
            raise TtsBridgeError(
                "ffmpeg nicht gefunden.\n"
                "Bitte ffmpeg installieren und im PATH verfügbar machen.\n"
                "Download: https://ffmpeg.org/download.html"
            )
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
