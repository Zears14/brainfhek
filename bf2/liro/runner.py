"""LIRO pass runner — executes an ordered list of passes on IR text."""

from __future__ import annotations

from typing import List

from bf2.liro import LIRO_REGISTRY, DEFAULT_LIRO_ORDER


def resolve_liro_spec(spec: str) -> List[str]:
    """Parse a ``-fliro=...`` specification into an ordered list of pass names.

    Rules:
        - ``""`` or ``None`` → DEFAULT_LIRO_ORDER (all defaults)
        - ``"N"`` (integer)  → [DEFAULT_LIRO_ORDER[N]]
        - ``"a,b,c"``       → ["a", "b", "c"]
    """
    if not spec:
        return list(DEFAULT_LIRO_ORDER)

    # Single integer → Nth default pass
    if spec.isdigit():
        idx = int(spec)
        if 0 <= idx < len(DEFAULT_LIRO_ORDER):
            return [DEFAULT_LIRO_ORDER[idx]]
        raise ValueError(
            f"LIRO index {idx} out of range (0..{len(DEFAULT_LIRO_ORDER) - 1})"
        )

    # Comma-separated names
    names = [n.strip() for n in spec.split(",") if n.strip()]
    for n in names:
        if n not in LIRO_REGISTRY:
            raise ValueError(
                f"Unknown LIRO pass '{n}'. "
                f"Available: {', '.join(sorted(LIRO_REGISTRY))}"
            )
    return names


def run_liros(ir: str, pass_names: List[str]) -> str:
    """Execute the given LIRO passes in order on the IR string.

    Each pass receives the full IR as a list of lines and returns a
    (possibly modified) list of lines.  Passes are independent and
    must not rely on execution order.
    """
    lines = ir.split("\n")
    for name in pass_names:
        cls = LIRO_REGISTRY[name]
        lines = cls().run(lines)
    return "\n".join(lines)
