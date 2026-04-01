import json
from typing import Callable, Optional

import requests

SATZZEICHEN = set('.!?:,;…"»')


def preprocess_text(text: str) -> str:
    """Fügt fehlende Satzzeichen am Zeilenende ein — läuft vor dem LLM-Aufruf."""
    lines = text.splitlines()
    result = []
    for line in lines:
        stripped = line.rstrip()
        if stripped and stripped[-1] not in SATZZEICHEN:
            line = stripped + '.'
        result.append(line)
    return '\n'.join(result)


class OllamaError(Exception):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    def __init__(self, model: str, available: list[str]):
        super().__init__(f"Modell '{model}' nicht gefunden")
        self.model = model
        self.available = available


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        cancelled_callback: Optional[Callable[[], bool]] = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": True,
        }
        if "qwen3" in model.lower():
            payload["options"] = {"think": False}

        try:
            response = requests.post(url, json=payload, stream=True, timeout=300)
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                "Ollama ist nicht erreichbar. Bitte starten Sie 'ollama serve'."
            )
        except requests.exceptions.Timeout:
            raise OllamaError("Zeitüberschreitung beim Verbinden mit Ollama.")

        if response.status_code == 404:
            available = self.list_models()
            raise OllamaModelNotFoundError(model, available)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise OllamaError(f"HTTP-Fehler: {e}")

        result_parts: list[str] = []
        try:
            for line in response.iter_lines():
                if cancelled_callback and cancelled_callback():
                    response.close()
                    return ""
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in data:
                    result_parts.append(data["response"])
                if data.get("done"):
                    break
        except requests.exceptions.ChunkedEncodingError as e:
            raise OllamaError(f"Verbindung während des Streamings unterbrochen: {e}")

        return "".join(result_parts)

    def unload_model(self, model: str) -> None:
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": "", "keep_alive": 0}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass  # Best effort — do not raise

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5)
            return True
        except Exception:
            return False
