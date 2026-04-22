"""Reactive update logic for Linked Segments."""

from __future__ import annotations
from typing import TYPE_CHECKING, Set

from llvmlite import ir

from bf2.core import ast as A
from bf2.backends.llvm.types import to_ir_type, align, Int32

if TYPE_CHECKING:
    from bf2.backends.llvm.emit_state import EmitState
    from bf2.backends.llvm.context import LLVMContext


from bf2.core.analysis import find_segment_deps


def emit_reactive_updates(st: EmitState, source_seg: str, index_v: ir.Value, visited: Set[str] | None = None) -> None:
    """Emit code to update all segments linked to the modified source segment at the given index."""
    if source_seg not in st.links:
        return
    
    if visited is None:
        visited = set()
    
    if source_seg in visited:
        return
    
    visited.add(source_seg)

    ctx = st.ctx
    # Prevent infinite recursion if segments are cross-linked
    # BF2 doesn't technically forbid it but it's dangerous.
    # We use a simple depth check or just trust the user for now.
    
    for linked_decl in st.links[source_seg]:
        _emit_link_update(st, linked_decl, index_v, visited)


def _emit_link_update(st: EmitState, decl: A.SegmentDecl, index_v: ir.Value, visited: Set[str]) -> None:
    """Update a specific linked segment at index_v."""
    if not decl.init:
        return

    ctx = st.ctx
    
    # Check if the link is still active (runtime)
    active_gv = st.link_active_gv.get(decl.name)
    if active_gv is not None:
        active = ctx.builder.load(active_gv, name=ctx.next_temp(f"link.active.{decl.name}"))
        with ctx.builder.if_then(active):
            _emit_link_update_logic(st, decl, index_v, visited)
    else:
        _emit_link_update_logic(st, decl, index_v, visited)

def _emit_link_update_logic(st: EmitState, decl: A.SegmentDecl, index_v: ir.Value, visited: Set[str]) -> None:
    ctx = st.ctx
    bound_expr = _bind_expr_to_index(decl.init, index_v)
    
    from bf2.backends.llvm.emit_expr import emit_expr
    from bf2.backends.llvm.emit_expr import _coerce
    
    val, ty = emit_expr(st, bound_expr, ctx)
    
    # Store into decl.name[index_v]
    target_ptr = st.seg_slots[decl.name]
    lty = to_ir_type(decl.elem_type)
    
    dest_ptr = ctx.builder.gep(target_ptr, [ir.Constant(Int32, 0), index_v], inbounds=True, name=ctx.next_temp("link.dest"))
    
    val = _coerce(st, val, ty, lty, ctx)
    ctx.builder.store(val, dest_ptr, align=align(lty))
    
    # Trigger watches on the linked segment update
    from bf2.backends.llvm.emit_watch import emit_maybe_watch
    emit_maybe_watch(st, decl.name, index_v)
    
    # Recurse: if this linked segment is also a source for others
    emit_reactive_updates(st, decl.name, index_v, visited)


def _bind_expr_to_index(expr: A.Expr, index_v: ir.Value) -> A.Expr:
    """Rewrite an expression so that all naked segment references use the provided index."""
    if isinstance(expr, A.Ident):
        # Treat naked IDENT as IDENT[index_v]
        # We wrap it in a RefExpr
        return A.RefExpr([expr.name, _SSAValeExpr(index_v)], expr.loc)
    elif isinstance(expr, A.RefExpr):
        # If it's already indexed, keep it? 
        # Actually, for linked segments like 'seg c = a + b', 
        # 'a' should mean 'a[i]'. If the user wrote 'a[0]', they might mean 'a[0]'.
        # But usually 'a' is what we want to bind.
        if len(expr.parts) == 1 and isinstance(expr.parts[0], str):
             return A.RefExpr([expr.parts[0], _SSAValeExpr(index_v)], expr.loc)
        return expr
    elif isinstance(expr, A.BinOp):
        return A.BinOp(expr.op, _bind_expr_to_index(expr.left, index_v), _bind_expr_to_index(expr.right, index_v), expr.loc)
    elif isinstance(expr, A.Unary):
        return A.Unary(expr.op, _bind_expr_to_index(expr.expr, index_v), expr.loc)
    elif isinstance(expr, A.Call):
        new_args = [_bind_expr_to_index(a, index_v) for a in expr.args]
        return A.Call(expr.name, new_args, expr.loc)
    return expr


class _SSAValeExpr(A.IntLit):
    """A hacky AST node to carry an LLVM SSA value through emit_expr."""
    def __init__(self, value: ir.Value):
        super().__init__(0) # Dummy value
        self.ssa_value = value

    def accept(self, visitor: any) -> any:
        # This will be handled specifically in emit_expr or similar
        return self.ssa_value
