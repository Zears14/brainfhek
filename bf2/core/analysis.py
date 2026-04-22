"""AST analysis utilities."""

from __future__ import annotations
from typing import Set

from bf2.core import ast as A

def find_segment_deps(expr: A.Expr) -> Set[str]:
    """Find all segment names used in an expression."""
    deps = set()
    if isinstance(expr, A.Ident):
        deps.add(expr.name)
    elif isinstance(expr, A.RefExpr):
        if expr.parts and isinstance(expr.parts[0], str):
            if expr.parts[0] != "@":
                deps.add(expr.parts[0])
            elif len(expr.parts) > 1 and isinstance(expr.parts[1], str):
                deps.add(expr.parts[1])
        # Also check indices for deps (unlikely but possible)
        for p in expr.parts:
            if isinstance(p, A.Expr):
                deps.update(find_segment_deps(p))
    elif isinstance(expr, A.BinOp):
        deps.update(find_segment_deps(expr.left))
        deps.update(find_segment_deps(expr.right))
    elif isinstance(expr, A.Unary):
        deps.update(find_segment_deps(expr.expr))
    elif isinstance(expr, A.Call):
        for arg in expr.args:
            deps.update(find_segment_deps(arg))
    return deps
