"""Diagnostic collector for parser and typechecker warnings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from bf2.core.errors import SourceLoc


@dataclass
class Diagnostic:
    """A single warning or note emitted during compilation."""

    level: str       # "warning", "note"
    code: str        # "ub", "unreachable", "shadow", "unused"
    message: str
    loc: Optional[SourceLoc] = None


class DiagnosticCollector:
    """Collects warnings and notes during compilation.

    Only warnings whose ``code`` is in the ``enabled`` set are recorded.
    """

    def __init__(self, enabled: Optional[Set[str]] = None) -> None:
        self.enabled: Set[str] = enabled or set()
        self.diagnostics: List[Diagnostic] = []

    def warn(self, code: str, msg: str, loc: Optional[SourceLoc] = None) -> None:
        """Record a warning if its code is enabled."""
        if code in self.enabled:
            self.diagnostics.append(Diagnostic("warning", code, msg, loc))

    def note(self, msg: str, loc: Optional[SourceLoc] = None) -> None:
        """Record an unconditional note."""
        self.diagnostics.append(Diagnostic("note", "", msg, loc))

    def has_warnings(self) -> bool:
        return any(d.level == "warning" for d in self.diagnostics)

    def format_all(self, source: str) -> str:
        """Format all diagnostics into a human-readable multi-line string."""
        if not self.diagnostics:
            return ""
        lines = source.splitlines()
        parts: List[str] = []
        for d in self.diagnostics:
            prefix = f"{d.level}"
            if d.code:
                prefix += f"[-W{d.code}]"
            if d.loc:
                ln = d.loc.line
                line_txt = lines[ln - 1] if 0 < ln <= len(lines) else ""
                parts.append(
                    f"{prefix} at line {d.loc.line}, col {d.loc.col}: {d.message}\n"
                    f"  {ln} | {line_txt}"
                )
            else:
                parts.append(f"{prefix}: {d.message}")
        return "\n".join(parts)
