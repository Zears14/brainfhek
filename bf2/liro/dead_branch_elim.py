"""LIRO: Dead Branch Elimination — remove unreachable basic blocks.

After static_watch_fold converts conditional branches to unconditional
ones, some labels may become unreachable.  This pass detects them and
removes the dead basic blocks.
"""

from __future__ import annotations

import re

from llvmlite import binding

from bf2.liro.base import LIROPass, register_liro

_BR_UNCOND_RE = re.compile(r"^\s+br\s+label\s+%(\S+)\s*(?:;.*)?$")

_BR_COND_RE = re.compile(
    r"^\s+br\s+i1\s+%\S+,\s*label\s+%(\S+),\s*label\s+%(\S+)\s*(?:;.*)?$"
)

_LABEL_RE = re.compile(r"^(\S+):\s*(?:;.*)?$")

_INVOKE_RE = re.compile(r"label\s+%")


@register_liro
class DeadBranchElim(LIROPass):
    name = "dead_branch_elim"
    description = "Remove basic blocks unreachable from any branch"

    def run(self, mod: binding.ModuleRef) -> binding.ModuleRef:
        ir_text = str(mod)
        lines = ir_text.split("\n")
        out: list[str] = []
        func_lines: list[str] = []
        in_func = False

        for line in lines:
            if line.startswith("define "):
                in_func = True
                func_lines = [line]
                continue

            if in_func:
                func_lines.append(line)
                if line.rstrip() == "}":
                    out.extend(self._process_function(func_lines))
                    in_func = False
                    func_lines = []
                continue

            out.append(line)

        return binding.parse_assembly("\n".join(out))

    def _process_function(self, lines: list[str]) -> list[str]:
        targets: set[str] = set()
        targets.add("entry")

        for line in lines:
            m = _BR_UNCOND_RE.match(line)
            if m:
                targets.add(m.group(1).strip('"'))
                continue
            m = _BR_COND_RE.match(line)
            if m:
                targets.add(m.group(1).strip('"'))
                targets.add(m.group(2).strip('"'))
                continue
            for phi_m in re.finditer(r"\[\s*[^,]+,\s*%(\S+)\s*\]", line):
                targets.add(phi_m.group(1).strip('"'))
            for inv_m in _INVOKE_RE.finditer(line):
                targets.add(inv_m.group(1).strip('"'))

        out: list[str] = []
        skip_block = False

        for line in lines:
            m = _LABEL_RE.match(line)
            if m:
                label_name = m.group(1).strip('"')
                if label_name not in targets:
                    skip_block = True
                    continue
                else:
                    skip_block = False

            if skip_block:
                if line.rstrip() == "}":
                    skip_block = False
                    out.append(line)
                    continue
                m2 = _LABEL_RE.match(line)
                if m2:
                    if m2.group(1).strip('"') in targets:
                        skip_block = False
                        out.append(line)
                    continue
                continue

            out.append(line)

        return out