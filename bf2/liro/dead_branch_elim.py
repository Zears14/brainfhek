"""LIRO: Dead Branch Elimination — remove unreachable basic blocks.

After static_watch_fold converts conditional branches to unconditional
ones, some labels may become unreachable.  This pass detects them and
removes the dead basic blocks.
"""

from __future__ import annotations

import re
from typing import List, Set

from bf2.liro.base import LIROPass, register_liro

# Matches: br label %target
_BR_UNCOND_RE = re.compile(r"^\s+br\s+label\s+%(\S+)\s*$")

# Matches: br i1 %cond, label %a, label %b
_BR_COND_RE = re.compile(
    r"^\s+br\s+i1\s+%\S+,\s*label\s+%(\S+),\s*label\s+%(\S+)\s*$"
)

# Matches a label definition: name:
_LABEL_RE = re.compile(r"^(\S+):\s*$")

# Matches: invoke ... to label %a unwind label %b
_INVOKE_RE = re.compile(r"label\s+%(\S+)")


@register_liro
class DeadBranchElim(LIROPass):
    name = "dead_branch_elim"
    description = "Remove basic blocks unreachable from any branch"

    def run(self, lines: List[str]) -> List[str]:
        # Process each function independently
        out: List[str] = []
        func_lines: List[str] = []
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

        return out

    def _process_function(self, lines: List[str]) -> List[str]:
        """Eliminate dead blocks within a single function."""
        # Collect all branch targets (labels that are jumped to)
        targets: Set[str] = set()
        targets.add("entry")  # entry is always reachable

        for line in lines:
            m = _BR_UNCOND_RE.match(line)
            if m:
                targets.add(m.group(1))
                continue
            m = _BR_COND_RE.match(line)
            if m:
                targets.add(m.group(1))
                targets.add(m.group(2))
                continue
            # Catch phi node references: [..., %label]
            for phi_m in re.finditer(r"\[\s*[^,]+,\s*%(\S+)\s*\]", line):
                targets.add(phi_m.group(1))
            # Catch invoke targets
            for inv_m in _INVOKE_RE.finditer(line):
                targets.add(inv_m.group(1))

        # Now filter: remove blocks whose label is not in targets
        out: List[str] = []
        skip_block = False

        for line in lines:
            m = _LABEL_RE.match(line)
            if m:
                label_name = m.group(1)
                if label_name not in targets:
                    skip_block = True
                    continue
                else:
                    skip_block = False

            if skip_block:
                # Check if we've hit the next label or end of function
                if line.rstrip() == "}":
                    skip_block = False
                    out.append(line)
                    continue
                m2 = _LABEL_RE.match(line)
                if m2:
                    # New label — check if this one is reachable
                    if m2.group(1) in targets:
                        skip_block = False
                        out.append(line)
                    continue
                continue

            out.append(line)

        return out
