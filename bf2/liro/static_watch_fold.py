"""LIRO: Static Watch Fold — constant-fold ``icmp eq i32 <const>, <const>`` patterns.

When the emitter generates a watch dispatch with two integer constants,
this pass resolves the comparison at IR level and replaces the branch
with an unconditional ``br`` to the appropriate target.
"""

from __future__ import annotations

import re
from typing import List

from bf2.liro.base import LIROPass, register_liro

# Matches: %reg = icmp eq i32 <const1>, <const2>
_ICMP_CONST_RE = re.compile(
    r"^\s+(%\S+)\s*=\s*icmp\s+eq\s+i32\s+(-?\d+),\s*(-?\d+)\s*$"
)

# Matches: br i1 %reg, label %a, label %b
_BR_I1_RE = re.compile(
    r"^\s+br\s+i1\s+(%\S+),\s*label\s+%(\S+),\s*label\s+%(\S+)\s*$"
)


@register_liro
class StaticWatchFold(LIROPass):
    name = "static_watch_fold"
    description = "Constant-fold icmp eq i32 <const>, <const> in watch dispatches"

    def run(self, lines: List[str]) -> List[str]:
        # First pass: find all constant icmp results
        const_regs: dict[str, bool] = {}
        for line in lines:
            m = _ICMP_CONST_RE.match(line)
            if m:
                reg, lhs, rhs = m.group(1), int(m.group(2)), int(m.group(3))
                const_regs[reg] = (lhs == rhs)

        if not const_regs:
            return lines

        # Second pass: rewrite
        out: List[str] = []
        skip_regs: set[str] = set()

        for line in lines:
            # Remove the constant icmp line itself
            m_icmp = _ICMP_CONST_RE.match(line)
            if m_icmp and m_icmp.group(1) in const_regs:
                skip_regs.add(m_icmp.group(1))
                continue

            # Rewrite conditional branches using folded results
            m_br = _BR_I1_RE.match(line)
            if m_br and m_br.group(1) in const_regs:
                reg = m_br.group(1)
                true_label = m_br.group(2)
                false_label = m_br.group(3)
                if const_regs[reg]:
                    out.append(f"  br label %{true_label}")
                else:
                    out.append(f"  br label %{false_label}")
                continue

            # Remove `and i1 %folded_reg, %other` → propagate
            and_match = re.match(
                r"^\s+(%\S+)\s*=\s*and\s+i1\s+(%\S+),\s*(%\S+)\s*$", line
            )
            if and_match:
                dest, op1, op2 = and_match.group(1), and_match.group(2), and_match.group(3)
                if op1 in const_regs:
                    if const_regs[op1]:
                        # true AND x → x, alias dest to op2
                        const_regs[dest] = const_regs.get(op2, True)
                        if op2 not in const_regs:
                            # op2 is dynamic, just rename
                            out.append(line)
                            continue
                    else:
                        # false AND x → false
                        const_regs[dest] = False
                    continue
                if op2 in const_regs:
                    if const_regs[op2]:
                        const_regs[dest] = const_regs.get(op1, True)
                        if op1 not in const_regs:
                            out.append(line)
                            continue
                    else:
                        const_regs[dest] = False
                    continue

            out.append(line)

        return out
