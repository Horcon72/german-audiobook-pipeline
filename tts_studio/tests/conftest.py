"""Fügt das tts_studio-Verzeichnis zum sys.path hinzu, damit pytest
die Pakete core.* und ui.* ohne Installation findet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
