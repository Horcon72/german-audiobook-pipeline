import re


class OcrScanner:
    """Regelbasierter OCR-Scanner – analysiert Text ohne LLM."""

    # Pipe, Bruchstrich und andere seltene Zeichen im deutschen Text
    _UNUSUAL_RE = re.compile(
        r"[|¦¡¿§©®™°±×÷†‡‰‱\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]"
    )

    # Buchstabe „b" am Ende eines Wortes unmittelbar vor Leerzeichen + Großbuchstabe
    # (häufige OCR-Fehlinterpretation von »)
    _SUSPICIOUS_B_RE = re.compile(
        r"(?<=[a-zäöüß])b(?=\s+[A-ZÄÖÜ\"\u201e\u00bb])"
    )

    def scan(self, text: str) -> dict:
        fundstellen: list[dict] = []

        fundstellen.extend(self._find_unpaired_quotes(text))
        fundstellen.extend(self._find_suspicious_chars(text))
        fundstellen.extend(self._find_unusual_chars(text))

        gesamt = len(fundstellen)
        if gesamt <= 5:
            qualitaet = "gut"
        elif gesamt <= 20:
            qualitaet = "mittel"
        else:
            qualitaet = "schlecht"

        return {
            "qualitaet": qualitaet,
            "gesamt_fehler": gesamt,
            "fehlertypen": {
                "unpaarige_anfuehrungszeichen": sum(
                    1 for f in fundstellen
                    if f["typ"] == "unpaarige_anfuehrungszeichen"
                ),
                "verdaechtige_zeichen": sum(
                    1 for f in fundstellen if f["typ"] == "verdaechtige_zeichen"
                ),
                "ungewoehnliche_kombinationen": sum(
                    1 for f in fundstellen
                    if f["typ"] == "ungewoehnliche_kombinationen"
                ),
            },
            "fundstellen": fundstellen,
        }

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _pos_to_line_col(text: str, pos: int) -> tuple[int, int]:
        before = text[:pos]
        line = before.count("\n") + 1
        col = pos - before.rfind("\n")
        return line, col

    @staticmethod
    def _get_display_context(text: str, pos: int, radius: int = 20) -> str:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
        ctx = text[start:end].replace("\n", "↵")
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        return prefix + ctx + suffix

    @staticmethod
    def _get_raw_context(text: str, pos: int, radius: int = 20) -> str:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
        return text[start:end]

    def _make_entry(self, text: str, pos: int, typ: str) -> dict:
        line, col = self._pos_to_line_col(text, pos)
        return {
            "zeile": line,
            "position": col,
            "typ": typ,
            "kontext": self._get_display_context(text, pos),
            "raw_kontext": self._get_raw_context(text, pos),
        }

    # ------------------------------------------------------------------
    # Erkennungslogik
    # ------------------------------------------------------------------

    def _find_unpaired_quotes(self, text: str) -> list[dict]:
        entries: list[dict] = []

        # --- Guillemets » « ---
        stack: list[int] = []
        for pos, ch in enumerate(text):
            if ch == "\u00bb":  # »
                stack.append(pos)
            elif ch == "\u00ab":  # «
                if stack:
                    stack.pop()
                else:
                    entries.append(
                        self._make_entry(text, pos, "unpaarige_anfuehrungszeichen")
                    )
        for pos in stack:
            entries.append(
                self._make_entry(text, pos, "unpaarige_anfuehrungszeichen")
            )

        # --- Typografische Anführungszeichen „ " ---
        stack = []
        for pos, ch in enumerate(text):
            if ch == "\u201e":  # „
                stack.append(pos)
            elif ch == "\u201c":  # "
                if stack:
                    stack.pop()
                else:
                    entries.append(
                        self._make_entry(text, pos, "unpaarige_anfuehrungszeichen")
                    )
        for pos in stack:
            entries.append(
                self._make_entry(text, pos, "unpaarige_anfuehrungszeichen")
            )

        return entries

    def _find_suspicious_chars(self, text: str) -> list[dict]:
        return [
            self._make_entry(text, m.start(), "verdaechtige_zeichen")
            for m in self._SUSPICIOUS_B_RE.finditer(text)
        ]

    def _find_unusual_chars(self, text: str) -> list[dict]:
        return [
            self._make_entry(text, m.start(), "ungewoehnliche_kombinationen")
            for m in self._UNUSUAL_RE.finditer(text)
        ]
