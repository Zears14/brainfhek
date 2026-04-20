"""Statement codegen: dispatch, control flow, cell ops, variable declarations."""

from __future__ import annotations

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.emit_expr import emit_expr, coerce
from bf2.backends.llvm.emit_mem import (
    gep, get_current_cell_ptr, emit_alloc, emit_free,
    emit_ptr_arith, emit_ptr_read, emit_ptr_write,
)
from bf2.backends.llvm.emit_io import emit_io
from bf2.backends.llvm.emit_watch import (
    try_static_seg_slot, emit_maybe_watch, emit_maybe_watch_current,
)
from bf2.backends.llvm.types import scalar_ty, align


def emit_stmt(st: EmitState, stmt: A.ASTNode) -> None:
    """Top-level statement dispatch."""
    if isinstance(stmt, A.SegmentStmt):
        _emit_local_seg(st, stmt.decl)
    elif isinstance(stmt, A.VarDecl):
        _emit_var_decl(st, stmt)
    elif isinstance(stmt, A.PtrDecl):
        _emit_ptr_decl(st, stmt)
    elif isinstance(stmt, A.AssignStmt):
        _emit_assign(st, stmt)
    elif isinstance(stmt, A.IfStmt):
        _emit_if(st, stmt)
    elif isinstance(stmt, A.LoopBF):
        _emit_loop_bf(st, stmt)
    elif isinstance(stmt, A.LoopCounted):
        _emit_loop_counted(st, stmt)
    elif isinstance(stmt, A.RetStmt):
        _emit_ret(st, stmt)
    elif isinstance(stmt, A.CallStmt):
        emit_expr(st, stmt.call, st.ctx)
    elif isinstance(stmt, A.MoveOp):
        _emit_move_op(st, stmt)
    elif isinstance(stmt, A.MoveRel):
        _emit_move_rel(st, stmt)
    elif isinstance(stmt, A.CellArith):
        _emit_cell_arith(st, stmt)
    elif isinstance(stmt, A.CellArithRef):
        _emit_cell_arith_ref(st, stmt)
    elif isinstance(stmt, A.CellAssignLit):
        _emit_cell_assign_lit(st, stmt)
    elif isinstance(stmt, A.IOStmt):
        emit_io(st, stmt)
    elif isinstance(stmt, A.LabelStmt):
        st.lines.append(f"{stmt.name}:")
    elif isinstance(stmt, A.JumpStmt):
        st.lines.append(f"  br label %{stmt.name}")
    elif isinstance(stmt, A.PtrArith):
        emit_ptr_arith(st, stmt)
    elif isinstance(stmt, A.PtrRead):
        emit_ptr_read(st, stmt.ptr, stmt.loc)
    elif isinstance(stmt, A.PtrWrite):
        emit_ptr_write(st, stmt)
    elif isinstance(stmt, A.AllocStmt):
        emit_alloc(st, stmt)
    elif isinstance(stmt, A.FreeStmt):
        emit_free(st, stmt.ptr)
    elif isinstance(stmt, A.RefExpr):
        emit_expr(st, stmt, st.ctx)
    elif isinstance(stmt, A.ExprStmt):
        emit_expr(st, stmt.expr, st.ctx)


# --- Local declarations ---


def _emit_local_seg(st: EmitState, d: A.SegmentDecl) -> None:
    lty = scalar_ty(d.elem_type)
    p = "%" + st.ctx.next_temp(f"lseg.{d.name}")
    st.alloca_lines.append(f"  {p} = alloca [{d.length} x {lty}], align {align(lty)}")
    st.ctx.locals[d.name] = (p, f"[{d.length} x {lty}]")
    st.seg_slots[d.name] = p


def _emit_var_decl(st: EmitState, d: A.VarDecl) -> None:
    lty = scalar_ty(d.ty)
    p = "%" + st.ctx.next_temp(f"v.{d.name}")
    st.alloca_lines.append(f"  {p} = alloca {lty}, align {align(lty)}")
    st.ctx.locals[d.name] = (p, lty)
    if d.init:
        rv, rt = emit_expr(st, d.init, st.ctx)
        rv = coerce(st, rv, rt, lty)
        st.lines.append(f"  store {lty} {rv}, ptr {p}, align {align(lty)}")


def _emit_ptr_decl(st: EmitState, d: A.PtrDecl) -> None:
    p = "%" + st.ctx.next_temp(f"ptr.{d.name}")
    st.alloca_lines.append(f"  {p} = alloca ptr, align 8")
    st.ctx.locals[d.name] = (p, "ptr")
    st.ctx.ptr_inner[d.name] = d.inner
    if d.init:
        rv, _ = emit_expr(st, d.init, st.ctx)
        st.lines.append(f"  store ptr {rv}, ptr {p}, align 8")


# --- Assignment ---


def _emit_assign(st: EmitState, s: A.AssignStmt) -> None:
    ptr, ty = gep(st, s.lhs, st.ctx)
    rv, rt = emit_expr(st, s.rhs, st.ctx)
    rv = coerce(st, rv, rt, ty)
    st.lines.append(f"  store {ty} {rv}, ptr {ptr}, align {align(ty)}")
    sk = try_static_seg_slot(st, s.lhs)
    if sk:
        emit_maybe_watch(st, sk[0], str(sk[1]))


# --- Control flow ---


def _emit_if(st: EmitState, s: A.IfStmt) -> None:
    cond_v = _emit_cond(st, s.cond)
    l_then = st.ctx.next_temp("if.then")
    l_else = st.ctx.next_temp("if.else")
    l_end = st.ctx.next_temp("if.end")

    st.lines.append(f"  br i1 {cond_v}, label %{l_then}, label %{l_else}")
    st.lines.append(f"{l_then}:")
    for s2 in s.then.stmts:
        emit_stmt(st, s2)
    st.lines.append(f"  br label %{l_end}")
    st.lines.append(f"{l_else}:")
    if s.els:
        for s2 in s.els.stmts:
            emit_stmt(st, s2)
    st.lines.append(f"  br label %{l_end}")
    st.lines.append(f"{l_end}:")


def _emit_loop_bf(st: EmitState, s: A.LoopBF) -> None:
    l_head = st.ctx.next_temp("loop.head")
    l_body = st.ctx.next_temp("loop.body")
    l_end = st.ctx.next_temp("loop.end")

    st.lines.append(f"  br label %{l_head}")
    st.lines.append(f"{l_head}:")
    p, ty = get_current_cell_ptr(st)
    t = "%" + st.ctx.next_temp("cv")
    st.lines.append(f"  {t} = load {ty}, ptr {p}, align {align(ty)}")
    is_zero = "%" + st.ctx.next_temp("iszero")
    if ty in ("float", "double"):
        st.lines.append(f"  {is_zero} = fcmp une {ty} {t}, 0.0")
    else:
        st.lines.append(f"  {is_zero} = icmp ne {ty} {t}, 0")
    st.lines.append(f"  br i1 {is_zero}, label %{l_body}, label %{l_end}")

    st.lines.append(f"{l_body}:")
    for s2 in s.body.stmts:
        emit_stmt(st, s2)
    st.lines.append(f"  br label %{l_head}")
    st.lines.append(f"{l_end}:")


def _emit_loop_counted(st: EmitState, s: A.LoopCounted) -> None:
    """Emit a counted loop using a phi node for the induction variable."""
    l_head = st.ctx.next_temp("loopc.head")
    l_body = st.ctx.next_temp("loopc.body")
    l_end = st.ctx.next_temp("loopc.end")

    # Record the label preceding the br so the phi can reference it
    pre_label = st.ctx.next_temp("loopc.pre")
    st.lines.append(f"  br label %{pre_label}")
    st.lines.append(f"{pre_label}:")

    st.lines.append(f"  br label %{l_head}")
    st.lines.append(f"{l_head}:")

    # Phi node for induction variable — avoids alloca+load+store round-trip
    iv = "%" + st.ctx.next_temp("iv")
    nv = "%" + st.ctx.next_temp("ne")
    st.lines.append(f"  {iv} = phi i32 [0, %{pre_label}], [{nv}, %{l_body}]")

    cond = "%" + st.ctx.next_temp("icond")
    st.lines.append(f"  {cond} = icmp slt i32 {iv}, {s.count}")
    st.lines.append(f"  br i1 {cond}, label %{l_body}, label %{l_end}")
    st.lines.append(f"{l_body}:")
    for s2 in s.body.stmts:
        emit_stmt(st, s2)
    st.lines.append(f"  {nv} = add nsw i32 {iv}, 1")
    st.lines.append(f"  br label %{l_head}")
    st.lines.append(f"{l_end}:")


def _emit_ret(st: EmitState, s: A.RetStmt) -> None:
    if not s.value:
        st.lines.append("  ret void")
    else:
        rv, rt = emit_expr(st, s.value, st.ctx)
        target_ty = st.ctx.ret_ty
        rv = coerce(st, rv, rt, target_ty)
        st.lines.append(f"  ret {target_ty} {rv}")


# --- Cursor / cell ops ---


def _emit_move_op(st: EmitState, s: A.MoveOp) -> None:
    ptr, ty = gep(st, s.target, st.ctx)
    st.ctx.cursor_type = ty
    st.lines.append(f"  store ptr {ptr}, ptr %__cseg, align 8")
    st.lines.append("  store i32 0, ptr %__cslot, align 4")


def _emit_move_rel(st: EmitState, s: A.MoveRel) -> None:
    cv = "%" + st.ctx.next_temp("idx")
    st.lines.append(f"  {cv} = load i32, ptr %__cslot, align 4")
    nv = "%" + st.ctx.next_temp("nidx")
    st.lines.append(f"  {nv} = add nsw i32 {cv}, {s.delta}")
    st.lines.append(f"  store i32 {nv}, ptr %__cslot, align 4")


def _emit_cell_arith(st: EmitState, s: A.CellArith) -> None:
    p, ty = get_current_cell_ptr(st)
    v = "%" + st.ctx.next_temp("val")
    st.lines.append(f"  {v} = load {ty}, ptr {p}, align {align(ty)}")
    amount = s.amount if s.amount is not None else 1
    nv = "%" + st.ctx.next_temp("nv")
    if ty in ("float", "double"):
        op = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}[s.op]
        st.lines.append(f"  {nv} = {op} {ty} {v}, {float(amount)}")
    else:
        op = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv"}[s.op]
        st.lines.append(f"  {nv} = {op} nsw {ty} {v}, {int(amount)}")
    st.lines.append(f"  store {ty} {nv}, ptr {p}, align {align(ty)}")
    emit_maybe_watch_current(st)


def _emit_cell_arith_ref(st: EmitState, s: A.CellArithRef) -> None:
    p, ty = gep(st, s.target, st.ctx)
    v = "%" + st.ctx.next_temp("val")
    st.lines.append(f"  {v} = load {ty}, ptr {p}, align {align(ty)}")
    amount = s.amount if s.amount is not None else 1
    nv = "%" + st.ctx.next_temp("nv")
    if ty in ("float", "double"):
        op = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}[s.op]
        st.lines.append(f"  {nv} = {op} {ty} {v}, {float(amount)}")
    else:
        op = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv"}[s.op]
        st.lines.append(f"  {nv} = {op} nsw {ty} {v}, {int(amount)}")
    st.lines.append(f"  store {ty} {nv}, ptr {p}, align {align(ty)}")
    sk = try_static_seg_slot(st, s.target)
    if sk:
        emit_maybe_watch(st, sk[0], str(sk[1]))


def _emit_cell_assign_lit(st: EmitState, s: A.CellAssignLit) -> None:
    p, ty = get_current_cell_ptr(st)
    if ty in ("float", "double"):
        st.lines.append(f"  store {ty} {float(s.value)}, ptr {p}, align {align(ty)}")
    elif ty == "i1":
        st.lines.append(f"  store i1 {'1' if s.value else '0'}, ptr {p}, align 1")
    else:
        st.lines.append(f"  store {ty} {int(s.value)}, ptr {p}, align {align(ty)}")
    emit_maybe_watch_current(st)


# --- Condition emission ---


def _emit_cond(st: EmitState, s: A.Cond) -> str:
    """Emit a cursor condition check and return the i1 SSA result."""
    p, ty = get_current_cell_ptr(st)
    v = "%" + st.ctx.next_temp("cv")
    st.lines.append(f"  {v} = load {ty}, ptr {p}, align {align(ty)}")
    t = "%" + st.ctx.next_temp("cond")
    is_fp = ty in ("float", "double")
    if is_fp:
        op_map = {
            ">0": "fcmp ogt", "<0": "fcmp olt", "==0": "fcmp oeq", "!=0": "fcmp une",
            ">N": "fcmp ogt", "<N": "fcmp olt", "==N": "fcmp oeq", "!=N": "fcmp une",
        }
        op = op_map[s.kind]
        imm = float(s.imm) if "N" in s.kind else 0.0
        st.lines.append(f"  {t} = {op} {ty} {v}, {imm}")
    else:
        op_map = {
            ">0": "icmp sgt", "<0": "icmp slt", "==0": "icmp eq", "!=0": "icmp ne",
            ">N": "icmp sgt", "<N": "icmp slt", "==N": "icmp eq", "!=N": "icmp ne",
        }
        op = op_map[s.kind]
        imm = s.imm if "N" in s.kind else 0
        st.lines.append(f"  {t} = {op} {ty} {v}, {imm}")
    return t
