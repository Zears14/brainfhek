from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceLoc:
    line: int
    col: int


class BF2Error(Exception):
    def __init__(self, msg: str, loc: SourceLoc | None = None):
        super().__init__(msg)
        self.msg = msg
        self.loc = loc


class BF2SyntaxError(BF2Error):
    pass


class BF2TypeError(BF2Error):
    pass


class BF2RuntimeError(BF2Error):
    pass


def format_error(
    err: BF2Error,
    source: str,
    title: str,
) -> str:
    loc = err.loc
    if loc is None:
        return f"{title}\n  {err.msg}"
    lines = source.splitlines()
    ln = loc.line
    line_txt = lines[ln - 1] if 0 < ln <= len(lines) else ""
    pad = len(str(ln))
    u = max(0, loc.col - 1)
    caret = " " * u + "^" + "~" * max(0, min(3, len(line_txt) - u - 1))
    return (
        f"{title} at line {loc.line}, col {loc.col}:\n"
        f"  {err.msg}\n\n"
        f"  {ln:>{pad}} | {line_txt}\n"
        f"  {' ' * pad} | {caret}"
    )
