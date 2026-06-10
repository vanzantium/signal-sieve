"""Language-pack contract for Signal Sieve's lexical risk layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


@dataclass
class LexiconPack:
    lang_code: str
    script: str
    certainty_markers: List[str] = field(default_factory=list)
    hedge_markers: List[str] = field(default_factory=list)
    hype_markers: List[str] = field(default_factory=list)
    attack_markers: List[str] = field(default_factory=list)
    evidence_markers: List[str] = field(default_factory=list)
    anonymous_markers: List[str] = field(default_factory=list)
    month_names: List[str] = field(default_factory=list)
    calibration: str = "starter"
    _compiled: Dict[str, Pattern] = field(default_factory=dict, repr=False)

    @staticmethod
    def _boundary(terms: List[str]) -> Pattern:
        if not terms:
            return re.compile(r"(?!x)x")
        parts = []
        for term in sorted(terms, key=len, reverse=True):
            escaped = re.escape(term)
            lead = r"\b" if term[:1].isalnum() else r"(?<!\w)"
            tail = r"\b" if term[-1:].isalnum() else r"(?!\w)"
            parts.append(f"{lead}{escaped}{tail}")
        return re.compile("(?:" + "|".join(parts) + ")", re.IGNORECASE)

    def compile(self) -> "LexiconPack":
        self._compiled = {
            "certainty": self._boundary(self.certainty_markers),
            "hedge": self._boundary(self.hedge_markers),
            "hype": self._boundary(self.hype_markers),
            "attack": self._boundary(self.attack_markers),
            "evidence": self._boundary(self.evidence_markers),
            "anonymous": self._boundary(self.anonymous_markers),
        }
        return self

    def find(self, kind: str, text: str) -> List[Tuple[int, int]]:
        if not self._compiled:
            self.compile()
        pattern = self._compiled.get(kind)
        if pattern is None:
            return []
        return [(match.start(), match.end()) for match in pattern.finditer(text)]

    def find_months(self, text: str) -> List[int]:
        """Return month positions only when the month is part of a dated phrase."""
        if not self.month_names:
            return []
        month = r"(?:%s)" % "|".join(
            sorted((re.escape(term) for term in self.month_names), key=len, reverse=True)
        )
        pattern = re.compile(
            rf"\b{month}\b\s+\d{{4}}"
            rf"|\b{month}\b\s+\d{{1,2}}(?:\s*,)?\s*\d{{4}}"
            rf"|\b\d{{1,2}}\s+{month}\b\s+\d{{4}}",
            re.IGNORECASE,
        )
        return [match.start() for match in pattern.finditer(text)]
