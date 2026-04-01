import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
PROFILES_FILE = DATA_DIR / "voice_profiles.json"


class VoiceProfileManager:
    """Verwaltet Stimmprofile als JSON unter data/voice_profiles.json."""

    def __init__(self):
        self._profiles: list[dict] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if PROFILES_FILE.exists():
            try:
                with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                    self._profiles = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._profiles = []
        else:
            self._profiles = []

    def _save(self):
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, ensure_ascii=False, indent=2)

    def get_all(self) -> list[dict]:
        return list(self._profiles)

    def get_by_id(self, id: str) -> Optional[dict]:
        for p in self._profiles:
            if p["id"] == id:
                return dict(p)
        return None

    def add(self, name: str, wav_path: str, txt_path: str) -> dict:
        profile = {
            "id": str(uuid.uuid4()),
            "name": name,
            "wav_path": wav_path,
            "txt_path": txt_path,
            "erstellt_am": datetime.now().isoformat(),
        }
        self._profiles.append(profile)
        self._save()
        return profile

    def delete(self, id: str) -> bool:
        for i, p in enumerate(self._profiles):
            if p["id"] == id:
                self._profiles.pop(i)
                self._save()
                return True
        return False
