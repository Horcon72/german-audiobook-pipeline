import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS: dict = {
    "ollama_url": "http://localhost:11434",
    "default_model": "gemma3:12b",
    "default_chunk_size": 2500,
    "recent_files": [],
    "cache_dir": "",
    "output_formats": ["mp3"],
    "tts_output_dir": "",
    "compose_file": "F:\\german-audiobook-pipeline\\tts\\docker-compose.yml",
}


class SettingsManager:
    def __init__(self):
        self._settings: dict = dict(DEFAULT_SETTINGS)
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self._settings.update(loaded)
            except Exception:
                pass

    def save(self):
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._settings, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default=None):
        return self._settings.get(key, default)

    def set(self, key: str, value):
        self._settings[key] = value
        self.save()

    def add_recent_file(self, path: str):
        recent: list = list(self._settings.get("recent_files", []))
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._settings["recent_files"] = recent[:5]
        self.save()

    def get_recent_files(self) -> list[str]:
        return list(self._settings.get("recent_files", []))
