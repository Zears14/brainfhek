from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Directory containing bundled standard headers (stdlib.bf2, …).
_PKG_ROOT = Path(__file__).resolve().parent.parent
_STDLIB_DIR = _PKG_ROOT / "stdlib"


class PreprocessError(Exception):
    pass


@dataclass
class PreprocessMeta:
    use_linux_stdlib: bool = False


class Preprocessor:
    """C-style subset preprocessor for Brainfhek."""

    _INCLUDE_RE = re.compile(
        r'^\s*#\s*include\s*([<"])([^>"]+)([>"])\s*(//.*)?$', re.MULTILINE
    )
    _DEFINE_RE = re.compile(
        r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.*?)\s*$"
    )

    def __init__(self, main_path: Path | None = None):
        self.base_dir = (main_path if main_path is not None else Path.cwd()).resolve().parent
        self.use_linux_stdlib = False

    def process_file(self, path: Path) -> tuple[str, PreprocessMeta]:
        text = path.read_text(encoding="utf-8")
        return self.process_text(text, path.resolve())

    def process_text(self, text: str, current_path: Path | None = None) -> tuple[str, PreprocessMeta]:
        base = current_path.parent if current_path else self.base_dir
        expanded = self._expand_includes(text, base, [current_path.resolve()] if current_path else [])
        stripped, defines = self._collect_defines(expanded)
        final = self._apply_defines(stripped, defines)
        return final, PreprocessMeta(use_linux_stdlib=self.use_linux_stdlib)

    def _resolve_include(self, name: str, relative_to: Path, angle: bool) -> Path:
        stem = name.strip()
        candidates = [stem] if stem.endswith(".bf2") else [f"{stem}.bf2", stem]

        search_paths = []
        if not angle:
            search_paths.append(relative_to)
        search_paths.extend([_STDLIB_DIR, _PKG_ROOT])

        for base in search_paths:
            for c in candidates:
                p = (base / c).resolve()
                if p.is_file():
                    return p
            if not stem.endswith(".bf2"):
                p = (base / stem / f"{stem}.bf2").resolve()
                if p.is_file():
                    return p

        raise PreprocessError(f'#include: file not found for {name!r}')

    def _expand_includes(self, text: str, relative_to: Path, stack: List[Path]) -> str:
        out: List[str] = []
        pos = 0
        for m in self._INCLUDE_RE.finditer(text):
            out.append(text[pos : m.start()])
            q1, path_part, q2 = m.group(1), m.group(2), m.group(3)
            
            if (q1 == "<" and q2 != ">") or (q1 == '"' and q2 != '"'):
                raise PreprocessError("mismatched #include delimiters")
                
            inc_path = self._resolve_include(path_part, relative_to, angle=(q1 == "<"))
            
            if inc_path.name == "stdlib.bf2" or path_part in ("stdlib", "stdlib.bf2"):
                self.use_linux_stdlib = True
                
            canon = inc_path.resolve()
            if canon in stack:
                raise PreprocessError(f"circular #include: {inc_path}")
                
            body = inc_path.read_text(encoding="utf-8")
            sub = self._expand_includes(body, inc_path.parent, stack + [canon])
            out.append(sub)
            pos = m.end()
            
        out.append(text[pos:])
        return "".join(out)

    def _collect_defines(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        defines: List[Tuple[str, str]] = []
        lines_out: List[str] = []
        for line in text.splitlines(keepends=True):
            m = self._DEFINE_RE.match(line)
            if m:
                name, raw = m.group(1), self._strip_line_comment(m.group(2))
                defines.append((name, raw))
                continue
            lines_out.append(line)
        return "".join(lines_out), defines

    def _apply_defines(self, text: str, defines: List[Tuple[str, str]]) -> str:
        out: List[str] = []
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("//"):
                out.append(line)
                continue
            out.append(self._apply_defines_to_line(line, defines))
        return "".join(out)

    def _apply_defines_to_line(self, line: str, defines: List[Tuple[str, str]]) -> str:
        text = line
        for _ in range(64):
            changed = False
            for name, val in defines:
                pat = re.compile(r"\b" + re.escape(name) + r"\b")
                new_text, n = pat.subn(val, text)
                if n:
                    changed = True
                    text = new_text
            if not changed:
                break
        return text

    def _strip_line_comment(self, value: str) -> str:
        s = value.strip()
        return s.split("//", 1)[0].strip() if "//" in s else s

def preprocess(text: str, *, main_path: Path | None = None) -> tuple[str, PreprocessMeta]:
    return Preprocessor(main_path).process_text(text)

def preprocess_path(path: Path) -> tuple[str, PreprocessMeta]:
    return Preprocessor(path).process_file(path)
