"""Reactor (watch) codegen with static constant folding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from llvmlite import ir

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.types import to_ir_type, Int32, Pointer

if TYPE_CHECKING:
    pass


def try_static_seg_slot(st: EmitState, r: A.RefExpr) -> Optional[Tuple[str, int]]:
    """Attempt to statically resolve a RefExpr to (segment_name, slot_index).

    Returns None if the reference is dynamic.
    """
    if len(r.parts) >= 1:
        head = r.parts[0]
        if head == "@":
            return (r.parts[1], 0)
        if head in st.global_segs:
            slot = 0
            if len(r.parts) > 1 and isinstance(r.parts[1], A.IntLit):
                slot = r.parts[1].value
            return (head, slot)
    return None


def emit_watch_fn(st: EmitState, idx: int, seg: str, slot: int, r: A.ReactorDef) -> None:
    """Emit a reactor function body ``@bf2.watch.N``."""
    from bf2.backends.llvm.emit_stmts import emit_stmt

    fn_ty = ir.FunctionType(ir.VoidType(), [])
    fn_name = f"bf2.watch.{idx}"
    fn = st.module.globals.get(fn_name)
    if fn is None:
        fn = ir.Function(st.module, fn_ty, name=fn_name)
    fn.attributes.add("nounwind")
    if fn.blocks:
        return
    entry_block = fn.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)

    seg_slot = st.seg_slots[seg]
    cursor_type = to_ir_type(st.global_segs[seg].elem_type)

    cseg_ptr = builder.alloca(Pointer, name="__cseg")
    cslot_ptr = builder.alloca(Int32, name="__cslot")

    slot_ptr = builder.gep(
        seg_slot,
        [ir.Constant(Int32, 0), ir.Constant(Int32, slot)],
        inbounds=True,
        name=f"watch.slot.{idx}",
    )
    slot_cast = builder.bitcast(slot_ptr, Pointer, name=f"watch.slot.cast.{idx}")
    builder.store(slot_cast, cseg_ptr, align=8)
    builder.store(ir.Constant(Int32, 0), cslot_ptr, align=4)

    st.ctx = LLVMContext(A.TypeRef("void"), builder)
    st.ctx.locals["__cseg"] = (cseg_ptr, Pointer)
    st.ctx.locals["__cslot"] = (cslot_ptr, Int32)
    st.ctx.cursor_type = cursor_type

    for stmt in r.body.stmts:
        emit_stmt(st, stmt)

    builder.ret_void()


def emit_maybe_watch(st: EmitState, seg: str, slot_v: ir.Value | str) -> None:
    """Emit reactor dispatch for a static or dynamic segment write."""
    ctx = st.ctx
    if ctx.builder.block.is_terminated:
        return
    
    if isinstance(slot_v, str):
        is_static_slot = slot_v.lstrip("-").isdigit()
        slot_v_ir = ir.Constant(Int32, int(slot_v)) if is_static_slot else None
    elif isinstance(slot_v, int):
        is_static_slot = True
        slot_v_ir = ir.Constant(Int32, slot_v)
    else:
        is_static_slot = False
        slot_v_ir = slot_v

    for i, (wseg, wslot, _) in enumerate(st.watches):
        if wseg != seg:
            continue

        if is_static_slot and int(slot_v) != wslot:
            continue

        # Prevent self-recursion: don't fire if we're already in this specific watch function
        watch_name = f"bf2.watch.{i}"
        in_this_watch = hasattr(ctx.builder, 'function') and ctx.builder.function.name == watch_name
        if in_this_watch:
            continue

        skip_block = ctx.builder.append_basic_block(name=ctx.next_temp(f"skip.watch.{i}"))
        fire_block = ctx.builder.append_basic_block(name=ctx.next_temp(f"fire.watch.{i}"))
        join_block = ctx.builder.append_basic_block(name=ctx.next_temp(f"join.watch.{i}"))

        if is_static_slot:
            mask_ptr = st.module.globals.get("bf2.watch.mask")
            if mask_ptr is None:
                mask_ptr = ir.GlobalVariable(st.module, ir.IntType(64), name="bf2.watch.mask")
                mask_ptr.initializer = ir.Constant(ir.IntType(64), 0)

            mask = ctx.builder.load(mask_ptr, align=8, name=ctx.next_temp("mask"))
            bit = ir.Constant(ir.IntType(64), 1 << i)
            already_firing = ctx.builder.and_(mask, bit, name=ctx.next_temp("already_firing"))
            can_fire = ctx.builder.icmp_signed("==", already_firing, ir.Constant(ir.IntType(64), 0), name=ctx.next_temp("can_fire"))
            ctx.builder.cbranch(can_fire, fire_block, skip_block)

            ctx.builder.position_at_end(fire_block)
            nmask = ctx.builder.or_(mask, bit, name=ctx.next_temp("nmask"))
            ctx.builder.store(nmask, mask_ptr, align=8)

            watch_name = f"bf2.watch.{i}"
            watch_fn = st.module.globals.get(watch_name)
            if watch_fn is None:
                watch_fn = ir.Function(st.module, ir.FunctionType(ir.VoidType(), []), name=watch_name)
            watch_fn.attributes.add("nounwind")
            ctx.builder.call(watch_fn, [])
            ctx.builder.store(mask, mask_ptr, align=8)
            ctx.builder.branch(join_block)

            ctx.builder.position_at_end(skip_block)
            ctx.builder.branch(join_block)

            ctx.builder.position_at_end(join_block)
        else:
            is_slot = ctx.builder.icmp_signed("==", slot_v_ir, ir.Constant(Int32, wslot), name=ctx.next_temp("is_slot"))

            mask_ptr = st.module.globals.get("bf2.watch.mask")
            if mask_ptr is None:
                mask_ptr = ir.GlobalVariable(st.module, ir.IntType(64), name="bf2.watch.mask")
                mask_ptr.initializer = ir.Constant(ir.IntType(64), 0)

            mask = ctx.builder.load(mask_ptr, align=8, name=ctx.next_temp("mask"))
            bit = ir.Constant(ir.IntType(64), 1 << i)
            already_firing = ctx.builder.and_(mask, bit, name=ctx.next_temp("already_firing"))
            not_firing = ctx.builder.icmp_signed("==", already_firing, ir.Constant(ir.IntType(64), 0), name=ctx.next_temp("not_firing"))
            must_fire = ctx.builder.and_(is_slot, not_firing, name=ctx.next_temp("must_fire"))
            ctx.builder.cbranch(must_fire, fire_block, skip_block)

            ctx.builder.position_at_end(fire_block)
            nmask = ctx.builder.or_(mask, bit, name=ctx.next_temp("nmask"))
            ctx.builder.store(nmask, mask_ptr, align=8)

            watch_name = f"bf2.watch.{i}"
            watch_fn = st.module.globals.get(watch_name)
            if watch_fn is None:
                watch_fn = ir.Function(st.module, ir.FunctionType(ir.VoidType(), []), name=watch_name)
            watch_fn.attributes.add("nounwind")
            ctx.builder.call(watch_fn, [])
            ctx.builder.store(mask, mask_ptr, align=8)
            ctx.builder.branch(join_block)

            ctx.builder.position_at_end(skip_block)
            ctx.builder.branch(join_block)

            ctx.builder.position_at_end(join_block)


def emit_maybe_watch_current(st: EmitState) -> None:
    """Emit reactor dispatch for the current dynamic cursor position."""
    if not st.watches:
        return

    ctx = st.ctx
    cseg_ptr, _ = ctx.locals["__cseg"]
    cslot_ptr, _ = ctx.locals["__cslot"]
    seg_ptr_val = ctx.builder.load(cseg_ptr, align=8, name=ctx.next_temp("cseg_ptr"))
    slot_v = ctx.builder.load(cslot_ptr, align=4, name=ctx.next_temp("cslot_val"))

    for i, (wseg, wslot, _) in enumerate(st.watches):
        wseg_ptr = st.seg_slots[wseg]
        watched_ptr = ctx.builder.gep(
            wseg_ptr,
            [ir.Constant(Int32, 0), ir.Constant(Int32, wslot)],
            inbounds=True,
            name=ctx.next_temp("watch.ptr"),
        )
        watched_cast = ctx.builder.bitcast(watched_ptr, Pointer, name=ctx.next_temp("watch.cast"))
        is_seg = ctx.builder.icmp_signed("==", seg_ptr_val, watched_cast, name=ctx.next_temp("is_seg"))
        is_slot = ctx.builder.icmp_signed("==", slot_v, ir.Constant(Int32, 0), name=ctx.next_temp("is_slot"))
        match = ctx.builder.and_(is_seg, is_slot, name=ctx.next_temp("match"))

        mask_ptr = st.module.globals.get("bf2.watch.mask")
        if mask_ptr is None:
            mask_ptr = ir.GlobalVariable(st.module, ir.IntType(64), name="bf2.watch.mask")
            mask_ptr.initializer = ir.Constant(ir.IntType(64), 0)

        mask = ctx.builder.load(mask_ptr, align=8, name=ctx.next_temp("mask"))
        bit = ir.Constant(ir.IntType(64), 1 << i)
        already_firing = ctx.builder.and_(mask, bit, name=ctx.next_temp("already_firing"))
        not_firing = ctx.builder.icmp_signed("==", already_firing, ir.Constant(ir.IntType(64), 0), name=ctx.next_temp("not_firing"))
        must_fire = ctx.builder.and_(match, not_firing, name=ctx.next_temp("must_fire"))
        
        # Prevent self-recursion
        watch_name = f"bf2.watch.{i}"
        in_this_watch = hasattr(ctx.builder, 'function') and ctx.builder.function.name == watch_name
        
        fire_block = ctx.builder.append_basic_block(name=ctx.next_temp(f"fire.watch.{i}"))
        skip_block = ctx.builder.append_basic_block(name=ctx.next_temp(f"skip.watch.{i}"))
        join_block = ctx.builder.append_basic_block(name=ctx.next_temp(f"join.watch.{i}"))

        # Combine conditions
        final_cond = ctx.builder.and_(must_fire, ctx.builder.not_(ir.Constant(ir.IntType(1), 1) if in_this_watch else ir.Constant(ir.IntType(1), 0)), name=ctx.next_temp("final_cond"))
        # Wait, that's complex. Let's just use Python 'if' if possible or a simple 'and'.
        
        if in_this_watch:
            ctx.builder.branch(skip_block)
        else:
            ctx.builder.cbranch(must_fire, fire_block, skip_block)

        ctx.builder.position_at_end(fire_block)
        nmask = ctx.builder.or_(mask, bit, name=ctx.next_temp("nmask"))
        ctx.builder.store(nmask, mask_ptr, align=8)

        watch_name = f"bf2.watch.{i}"
        watch_fn = st.module.globals.get(watch_name)
        if watch_fn is None:
            watch_fn = ir.Function(st.module, ir.FunctionType(ir.VoidType(), []), name=watch_name)
        watch_fn.attributes.add("nounwind")
        ctx.builder.call(watch_fn, [])
        ctx.builder.store(seg_ptr_val, cseg_ptr, align=8)
        ctx.builder.store(slot_v, cslot_ptr, align=4)
        ctx.builder.store(mask, mask_ptr, align=8)
        ctx.builder.branch(join_block)

        ctx.builder.position_at_end(skip_block)
        ctx.builder.branch(join_block)

        ctx.builder.position_at_end(join_block)
