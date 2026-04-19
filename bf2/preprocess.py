"""
C-style preprocessor subset: #include and #define.

Linux-focused: resolving #include "stdlib" loads the bundled bf2/stdlib/stdlib.bf2
and records that the Linux libc stdlib declarations should be emitted in LLVM IR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Directory containing bundled standard headers (stdlib.bf2, …).
_PKG_ROOT = Path(__file__).resolve().parent
_STDLIB_DIR = _PKG_ROOT / "stdlib"


class PreprocessError(Exception):
    pass


@dataclass
class PreprocessMeta:
    use_linux_stdlib: bool = False


_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*([<"])([^>"]+)([>"])\s*(//.*)?$', re.MULTILINE
)
_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.*?)\s*$"
)


def _strip_line_comment(value: str) -> str:
    s = value.strip()
    if "//" in s:
        s = s.split("//", 1)[0].strip()
    return s


def _resolve_include(name: str, relative_to: Path, angle: bool) -> Path:
    """Locate an included file. ``angle`` is True for #include <...>."""
    stem = name.strip()
    if not stem.endswith(".bf2"):
        candidates: list[str] = [f"{stem}.bf2", stem]
    else:
        candidates = [stem]

    search: list[Path] = []
    if not angle:
        search.append(relative_to)
    search.append(_STDLIB_DIR)
    search.append(_PKG_ROOT)

    tried: list[str] = []
    for base in search:
        for c in candidates:
            p = (base / c).resolve()
            tried.append(str(p))
            if p.is_file():
                return p
        # e.g. stdlib/stdlib.bf2 when requesting "stdlib"
        if not stem.endswith(".bf2"):
            p = (base / stem / f"{stem}.bf2").resolve()
            tried.append(str(p))
            if p.is_file():
                return p

    raise PreprocessError(f'#include: file not found for {name!r} (tried search near {relative_to})')


def _expand_includes_once(text: str, relative_to: Path, stack: list[Path]) -> tuple[str, bool]:
    """Expand #include directives one level; included text is processed recursively."""

    use_stdlib = False
    out: list[str] = []
    pos = 0
    for m in _INCLUDE_RE.finditer(text):
        out.append(text[pos : m.start()])
        q1, path_part, q2 = m.group(1), m.group(2), m.group(3)
        if (q1 == "<" and q2 != ">") or (q1 == '"' and q2 != '"'):
            raise PreprocessError("mismatched #include delimiters", m.start())
        angle = q1 == "<"
        inc_path = _resolve_include(path_part, relative_to, angle=angle)
        if inc_path.name == "stdlib.bf2" or path_part in ("stdlib", "stdlib.bf2"):
            use_stdlib = True
        canon = inc_path.resolve()
        if canon in stack:
            raise PreprocessError(f"circular #include: {inc_path}")
        body = inc_path.read_text(encoding="utf-8")
        sub, u = _expand_includes_once(body, inc_path.parent, stack + [canon])
        use_stdlib = use_stdlib or u
        out.append(sub)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), use_stdlib


def _collect_defines(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove #define lines and return ordered (name, value) pairs."""
    defines: list[tuple[str, str]] = []
    lines_out: list[str] = []
    for line in text.splitlines(keepends=True):
        m = _DEFINE_RE.match(line)
        if m:
            name, raw = m.group(1), _strip_line_comment(m.group(2))
            defines.append((name, raw))
            continue
        lines_out.append(line)
    return "".join(lines_out), defines


def _apply_defines_to_line(line: str, defines: Iterable[tuple[str, str]]) -> str:
    """Apply macro substitution to one line (already known not to be a // comment)."""
    text = line
    dlist = list(defines)
    for _ in range(64):
        changed = False
        for name, val in dlist:
            pat = re.compile(r"\b" + re.escape(name) + r"\b")
            new_text, n = pat.subn(val, text)
            if n:
                changed = True
                text = new_text
        if not changed:
            break
    return text


def _apply_defines(text: str, defines: Iterable[tuple[str, str]]) -> str:
    """Substitute macros; leaves ``//`` full-line comments untouched."""
    dlist = list(defines)
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        core = line.lstrip()
        if core.startswith("//"):
            out.append(line)
            continue
        out.append(_apply_defines_to_line(line, dlist))
    return "".join(out)


def preprocess(text: str, *, main_path: Path | None = None) -> tuple[str, PreprocessMeta]:
    """
    Expand #includes, apply #defines. ``main_path`` is the source file path used to
    resolve quoted includes; defaults to the current working directory.
    """
    base = (main_path if main_path is not None else Path.cwd()).resolve().parent
    expanded, use_std = _expand_includes_once(text, base, [])
    stripped, defs = _collect_defines(expanded)
    final = _apply_defines(stripped, defs)
    return final, PreprocessMeta(use_linux_stdlib=use_std)


def preprocess_path(path: Path) -> tuple[str, PreprocessMeta]:
    """Read *path* and run :func:`preprocess` with ``main_path`` set."""
    src = path.read_text(encoding="utf-8")
    return preprocess(src, main_path=path.resolve())
