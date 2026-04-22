"""Statement codegen: dispatch, control flow, cell ops, variable declarations."""

from __future__ import annotations

from llvmlite import ir

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.emit_expr import emit_expr
from bf2.backends.llvm.emit_mem import (
    gep, get_current_cell_ptr, emit_alloc, emit_free,
    emit_ptr_arith, emit_ptr_read, emit_ptr_write, resolve_write_target,
)
from bf2.backends.llvm.emit_io import emit_io
from bf2.backends.llvm.emit_watch import (
    try_static_seg_slot, emit_maybe_watch, emit_maybe_watch_current,
)
from bf2.backends.llvm.emit_reactive import emit_reactive_updates
from bf2.backends.llvm.types import to_ir_type, align, Int1, Int32, Pointer


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
    elif isinstance(stmt, A.LoadOp):
        _emit_load_op(st, stmt)
    elif isinstance(stmt, A.StoreOp):
        _emit_store_op(st, stmt)
    elif isinstance(stmt, A.SwapOp):
        _emit_swap_op(st, stmt)
    elif isinstance(stmt, A.IOStmt):
        emit_io(st, stmt)
    elif isinstance(stmt, A.LabelStmt):
        safe_name = stmt.name.replace('.', '_')
        block = st.ctx.blocks.get(safe_name)
        if block is None:
            block = st.ctx.builder.append_basic_block(name=safe_name)
            st.ctx.blocks[safe_name] = block
        st.ctx.builder.position_at_end(block)
    elif isinstance(stmt, A.JumpStmt):
        safe_name = stmt.name.replace('.', '_')
        target_block = st.ctx.blocks.get(safe_name)
        if target_block is None:
            target_block = st.ctx.builder.append_basic_block(name=safe_name)
            st.ctx.blocks[safe_name] = target_block
        st.ctx.builder.branch(target_block)
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


def _emit_local_seg(st: EmitState, d: A.SegmentDecl) -> None:
    ctx = st.ctx
    lty = to_ir_type(d.elem_type)
    arr_ty = ir.ArrayType(lty, d.length)
    ptr = ctx.hoist_alloca(arr_ty, name=ctx.next_temp(f"lseg.{d.name}"))
    ctx.locals[d.name] = (ptr, arr_ty)
    st.seg_slots[d.name] = ptr
    if d.length > 0:
        ctx.builder.store(ir.Constant(arr_ty, None), ptr, align=align(lty))


def _emit_var_decl(st: EmitState, d: A.VarDecl) -> None:
    ctx = st.ctx
    lty = to_ir_type(d.ty)
    ptr = ctx.hoist_alloca(lty, name=ctx.next_temp(f"v.{d.name}"))
    ctx.locals[d.name] = (ptr, lty)
    if d.init:
        rv, rt = emit_expr(st, d.init, ctx)
        from bf2.backends.llvm.emit_expr import _coerce
        rv = _coerce(st, rv, rt, lty, ctx)
        ctx.builder.store(rv, ptr, align=align(lty))


def _emit_ptr_decl(st: EmitState, d: A.PtrDecl) -> None:
    ctx = st.ctx
    ptr_type = to_ir_type(A.TypeRef("ptr", d.inner))
    ptr = ctx.hoist_alloca(ptr_type, name=ctx.next_temp(f"ptr.{d.name}"))
    ctx.locals[d.name] = (ptr, ptr_type)
    ctx.ptr_inner[d.name] = d.inner
    if d.init:
        rv, rt = emit_expr(st, d.init, ctx)
        if not isinstance(rt, ir.PointerType):
            raise TypeError(f"pointer init for {d.name} must be a pointer, got {rt}")
        if rv.type != ptr_type:
            rv = ctx.builder.bitcast(rv, ptr_type, name=ctx.next_temp("bc"))
        ctx.builder.store(rv, ptr)


def _emit_assign(st: EmitState, s: A.AssignStmt) -> None:
    ctx = st.ctx
    ptr, ty, seg_n, idx_v = resolve_write_target(st, s.lhs, ctx)
    rv, rt = emit_expr(st, s.rhs, ctx)
    from bf2.backends.llvm.emit_expr import _coerce
    rv = _coerce(st, rv, rt, ty, ctx)
    ctx.builder.store(rv, ptr, align=align(ty))
    
    if seg_n and idx_v:
        emit_reactive_updates(st, seg_n, idx_v)

    sk = try_static_seg_slot(st, s.lhs)
    if sk and not ctx.builder.block.is_terminated:
        emit_maybe_watch(st, sk[0], sk[1])


def _emit_if(st: EmitState, s: A.IfStmt) -> None:
    ctx = st.ctx
    cond_v = _emit_cond(st, s.cond)

    then_block = ctx.builder.append_basic_block(name=ctx.next_temp("if.then"))
    else_block = ctx.builder.append_basic_block(name=ctx.next_temp("if.else"))
    end_block = ctx.builder.append_basic_block(name=ctx.next_temp("if.end"))

    ctx.builder.cbranch(cond_v, then_block, else_block)

    ctx.builder.position_at_end(then_block)
    for s2 in s.then.stmts:
        emit_stmt(st, s2)
    if not ctx.builder.block.terminator:
        ctx.builder.branch(end_block)

    ctx.builder.position_at_end(else_block)
    if s.els:
        for s2 in s.els.stmts:
            emit_stmt(st, s2)
    if not ctx.builder.block.terminator:
        ctx.builder.branch(end_block)

    ctx.builder.position_at_end(end_block)


def _emit_loop_bf(st: EmitState, s: A.LoopBF) -> None:
    ctx = st.ctx
    head_block = ctx.builder.append_basic_block(name=ctx.next_temp("loop.head"))
    body_block = ctx.builder.append_basic_block(name=ctx.next_temp("loop.body"))
    end_block = ctx.builder.append_basic_block(name=ctx.next_temp("loop.end"))

    ctx.builder.branch(head_block)

    ctx.builder.position_at_end(head_block)
    p, ty = get_current_cell_ptr(st)
    t = ctx.builder.load(p, align=align(ty), name=ctx.next_temp("cv"))
    if isinstance(ty, (ir.FloatType, ir.DoubleType)):
        is_zero = ctx.builder.fcmp_unordered("!=", t, ir.Constant(ty, 0.0), name=ctx.next_temp("iszero"))
    else:
        is_zero = ctx.builder.icmp_signed("!=", t, ir.Constant(ty, 0), name=ctx.next_temp("iszero"))
    ctx.builder.cbranch(is_zero, body_block, end_block)

    ctx.builder.position_at_end(body_block)
    for s2 in s.body.stmts:
        emit_stmt(st, s2)
    if not ctx.builder.block.is_terminated:
        br = ctx.builder.branch(head_block)
        loop_md = st.module.add_metadata([])
        loop_md.operands = [loop_md]
        br.set_metadata("llvm.loop", loop_md)

    ctx.builder.position_at_end(end_block)


def _emit_loop_counted(st: EmitState, s: A.LoopCounted) -> None:
    """Emit a counted loop using a phi node for the induction variable."""
    ctx = st.ctx
    pre_block = ctx.builder.append_basic_block(name=ctx.next_temp("loopc.pre"))
    head_block = ctx.builder.append_basic_block(name=ctx.next_temp("loopc.head"))
    body_block = ctx.builder.append_basic_block(name=ctx.next_temp("loopc.body"))
    end_block = ctx.builder.append_basic_block(name=ctx.next_temp("loopc.end"))

    ctx.builder.branch(pre_block)

    ctx.builder.position_at_end(pre_block)
    ctx.builder.branch(head_block)

    ctx.builder.position_at_end(head_block)
    phi = ctx.builder.phi(ir.IntType(64), name=ctx.next_temp("iv"))
    phi.add_incoming(ir.Constant(ir.IntType(64), 0), pre_block)

    cond = ctx.builder.icmp_signed("<", phi, ir.Constant(ir.IntType(64), s.count), name=ctx.next_temp("icond"))
    ctx.builder.cbranch(cond, body_block, end_block)

    ctx.builder.position_at_end(body_block)
    for s2 in s.body.stmts:
        emit_stmt(st, s2)
    if not ctx.builder.block.is_terminated:
        nv = ctx.builder.add(phi, ir.Constant(ir.IntType(64), 1), name=ctx.next_temp("ne"), flags=["nsw"])
        br = ctx.builder.branch(head_block)
        loop_md = st.module.add_metadata([])
        loop_md.operands = [loop_md]
        br.set_metadata("llvm.loop", loop_md)
        phi.add_incoming(nv, body_block)

    ctx.builder.position_at_end(end_block)


def _emit_ret(st: EmitState, s: A.RetStmt) -> None:
    ctx = st.ctx
    if ctx.builder.block.is_terminated:
        return
    if not s.value:
        ctx.builder.ret_void()
    else:
        rv, rt = emit_expr(st, s.value, ctx)
        from bf2.backends.llvm.emit_expr import _coerce
        rv = _coerce(st, rv, rt, ctx.ret_ty, ctx)
        ctx.builder.ret(rv)


def _emit_move_op(st: EmitState, s: A.MoveOp) -> None:
    ctx = st.ctx
    ptr, ty, seg_n, idx_v = resolve_write_target(st, s.target, ctx)
    ctx.cursor_type = ty
    ctx.cursor_seg_name = seg_n
    ctx.cursor_index_v = idx_v
    if "__cseg" in ctx.locals:
        cseg_ptr, _ = ctx.locals["__cseg"]
    else:
        cseg_ptr = ctx.hoist_alloca(Pointer, name="__cseg")
        ctx.locals["__cseg"] = (cseg_ptr, Pointer)
    if "__cslot" in ctx.locals:
        cslot_ptr, _ = ctx.locals["__cslot"]
    else:
        cslot_ptr = ctx.hoist_alloca(Int32, name="__cslot")
        ctx.locals["__cslot"] = (cslot_ptr, Int32)
    ptr_cast = ctx.builder.bitcast(ptr, Pointer, name="cseg_cast")
    ctx.builder.store(ptr_cast, cseg_ptr)
    ctx.builder.store(ir.Constant(Int32, 0), cslot_ptr)


def _emit_move_rel(st: EmitState, s: A.MoveRel) -> None:
    ctx = st.ctx
    if "__cslot" in ctx.locals:
        cslot_ptr, _ = ctx.locals["__cslot"]
    else:
        cslot_ptr = ctx.hoist_alloca(Int32, name="__cslot")
        ctx.locals["__cslot"] = (cslot_ptr, Int32)
    cv = ctx.builder.load(cslot_ptr, name=ctx.next_temp("idx"))
    nv = ctx.builder.add(cv, ir.Constant(Int32, s.delta), name=ctx.next_temp("nidx"), flags=["nsw"])
    ctx.builder.store(nv, cslot_ptr)
    if ctx.cursor_seg_name and ctx.cursor_index_v is not None:
        ctx.cursor_index_v = ctx.builder.add(ctx.cursor_index_v, ir.Constant(Int32, s.delta), name=ctx.next_temp("rel_idx"))


def _emit_cell_arith(st: EmitState, s: A.CellArith) -> None:
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    v = ctx.builder.load(p, align=align(ty), name=ctx.next_temp("val"))
    amount = s.amount if s.amount is not None else 1
    if isinstance(ty, (ir.FloatType, ir.DoubleType)):
        fp_ops = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}
        op = fp_ops[s.op]
        result = getattr(ctx.builder, op)(v, ir.Constant(ty, float(amount)), name=ctx.next_temp("nv"))
    else:
        int_ops = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv"}
        op = int_ops[s.op]
        flags = ["nsw"] if s.op in ("+", "-", "*") else []
        result = ctx.builder.add(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags) if s.op == "+" else (
            ctx.builder.sub(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags) if s.op == "-" else (
            ctx.builder.mul(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags) if s.op == "*" else
            ctx.builder.sdiv(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags)
        ))
    ctx.builder.store(result, p, align=align(ty))
    if ctx.cursor_seg_name and ctx.cursor_index_v is not None:
        emit_reactive_updates(st, ctx.cursor_seg_name, ctx.cursor_index_v)
    emit_maybe_watch_current(st)


def _emit_cell_arith_ref(st: EmitState, s: A.CellArithRef) -> None:
    ctx = st.ctx
    p, ty, seg_n, idx_v = resolve_write_target(st, s.target, ctx)
    v = ctx.builder.load(p, align=align(ty), name=ctx.next_temp("val"))
    amount = s.amount if s.amount is not None else 1
    if isinstance(ty, (ir.FloatType, ir.DoubleType)):
        fp_ops = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}
        op = fp_ops[s.op]
        result = getattr(ctx.builder, op)(v, ir.Constant(ty, float(amount)), name=ctx.next_temp("nv"))
    else:
        int_ops = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv"}
        op = int_ops[s.op]
        flags = ["nsw"] if s.op in ("+", "-", "*") else []
        result = ctx.builder.add(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags) if s.op == "+" else (
            ctx.builder.sub(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags) if s.op == "-" else (
            ctx.builder.mul(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags) if s.op == "*" else
            ctx.builder.sdiv(v, ir.Constant(ty, int(amount)), name=ctx.next_temp("nv"), flags=flags)
        ))
    ctx.builder.store(result, p, align=align(ty))
    if seg_n and idx_v:
        emit_reactive_updates(st, seg_n, idx_v)
    sk = try_static_seg_slot(st, s.target)
    if sk:
        emit_maybe_watch(st, sk[0], str(sk[1]))


def _emit_cell_assign_lit(st: EmitState, s: A.CellAssignLit) -> None:
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    if isinstance(ty, (ir.FloatType, ir.DoubleType)):
        ctx.builder.store(ir.Constant(ty, float(s.value)), p, align=align(ty))
    elif ty == Int1:
        ctx.builder.store(ir.Constant(Int1, 1 if s.value else 0), p, align=1)
    else:
        ctx.builder.store(ir.Constant(ty, int(s.value)), p, align=align(ty))
    if ctx.cursor_seg_name and ctx.cursor_index_v is not None:
        emit_reactive_updates(st, ctx.cursor_seg_name, ctx.cursor_index_v)
    emit_maybe_watch_current(st)


def _emit_load_op(st: EmitState, s: A.LoadOp) -> None:
    ctx = st.ctx
    src_ptr, src_ty = gep(st, s.src, ctx)
    src_val = ctx.builder.load(src_ptr, align=align(src_ty), name=ctx.next_temp("load"))
    dst_ptr, dst_ty = get_current_cell_ptr(st)
    from bf2.backends.llvm.emit_expr import _coerce
    src_val = _coerce(st, src_val, src_ty, dst_ty, ctx)
    ctx.builder.store(src_val, dst_ptr, align=align(dst_ty))


def _emit_store_op(st: EmitState, s: A.StoreOp) -> None:
    ctx = st.ctx
    src_ptr, src_ty = get_current_cell_ptr(st)
    src_val = ctx.builder.load(src_ptr, align=align(src_ty), name=ctx.next_temp("storev"))
    dst_ptr, dst_ty, seg_n, idx_v = resolve_write_target(st, s.dst, ctx)
    from bf2.backends.llvm.emit_expr import _coerce
    src_val = _coerce(st, src_val, src_ty, dst_ty, ctx)
    ctx.builder.store(src_val, dst_ptr, align=align(dst_ty))
    if seg_n and idx_v:
        emit_reactive_updates(st, seg_n, idx_v)
    sk = try_static_seg_slot(st, s.dst)
    if sk and not ctx.builder.block.is_terminated:
        emit_maybe_watch(st, sk[0], sk[1])


def _emit_swap_op(st: EmitState, s: A.SwapOp) -> None:
    ctx = st.ctx
    other_ptr, other_ty, seg_n, idx_v = resolve_write_target(st, s.other, ctx)
    other_val = ctx.builder.load(other_ptr, align=align(other_ty), name=ctx.next_temp("swap.other"))
    cell_ptr, cell_ty = get_current_cell_ptr(st)
    cell_val = ctx.builder.load(cell_ptr, align=align(cell_ty), name=ctx.next_temp("swap.cell"))
    from bf2.backends.llvm.emit_expr import _coerce
    store_other = _coerce(st, cell_val, cell_ty, other_ty, ctx)
    store_cell = _coerce(st, other_val, other_ty, cell_ty, ctx)
    ctx.builder.store(store_other, other_ptr, align=align(other_ty))
    ctx.builder.store(store_cell, cell_ptr, align=align(cell_ty))
    
    if seg_n and idx_v:
        emit_reactive_updates(st, seg_n, idx_v)
    if ctx.cursor_seg_name and ctx.cursor_index_v is not None:
        emit_reactive_updates(st, ctx.cursor_seg_name, ctx.cursor_index_v)

    sk = try_static_seg_slot(st, s.other)
    if sk and not ctx.builder.block.is_terminated:
        emit_maybe_watch(st, sk[0], sk[1])


def _emit_cond(st: EmitState, s: A.Cond) -> ir.Value:
    """Emit a cursor condition check and return the i1 SSA result."""
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    v = ctx.builder.load(p, align=align(ty), name=ctx.next_temp("cv"))
    is_fp = isinstance(ty, (ir.FloatType, ir.DoubleType))
    if is_fp:
        op_map = {
            ">0": "ogt", "<0": "olt", "==0": "oeq", "!=0": "une",
            ">N": "ogt", "<N": "olt", "==N": "oeq", "!=N": "une",
        }
        op = op_map[s.kind]
        imm = float(s.imm) if "N" in s.kind else 0.0
        return ctx.builder.fcmp_unordered(op, v, ir.Constant(ty, imm), name=ctx.next_temp("cond"))
    else:
        op_map = {
            ">0": ">", "<0": "<", "==0": "==", "!=0": "!=",
            ">N": ">", "<N": "<", "==N": "==", "!=N": "!=",
        }
        op = op_map[s.kind]
        imm = s.imm if "N" in s.kind else 0
        return ctx.builder.icmp_signed(op, v, ir.Constant(ty, imm), name=ctx.next_temp("cond"))
