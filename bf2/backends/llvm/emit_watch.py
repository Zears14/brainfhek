"""Reactor (watch) codegen with static constant folding."""

from __future__ import annotations

from typing import Optional, Tuple

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.types import scalar_ty


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
    from bf2.backends.llvm.context import LLVMContext
    from bf2.backends.llvm.emit_stmts import emit_stmt

    st.alloca_lines = []
    st.ctx = LLVMContext("void")
    st.lines.append(f"define void @bf2.watch.{idx}() #0 {{")
    st.lines.append("entry:")
    insertion_point = len(st.lines)
    # Reactors run with cursor at watched slot
    p = st.seg_slots[seg]
    st.alloca_lines.append("  %__cseg = alloca ptr, align 8")
    st.alloca_lines.append("  %__cslot = alloca i32, align 4")
    st.lines.append(f"  store ptr {p}, ptr %__cseg, align 8")
    st.lines.append(f"  store i32 {slot}, ptr %__cslot, align 4")
    st.ctx.cursor_type = scalar_ty(st.global_segs[seg].elem_type)
    for stmt in r.body.stmts:
        emit_stmt(st, stmt)
    st.lines[insertion_point:insertion_point] = st.alloca_lines
    st.lines.append("  ret void")
    st.lines.append("}")


def emit_maybe_watch(st: EmitState, seg: str, slot_ssa: str) -> None:
    """Emit reactor dispatch for a static segment write.

    **Static constant folding**: if ``slot_ssa`` is a plain integer string,
    we compare it against each watch's slot at Python compile time.  When
    they don't match, we emit zero IR (the reactor can never fire).
    """
    is_static_slot = slot_ssa.lstrip("-").isdigit()

    for i, (wseg, wslot, _) in enumerate(st.watches):
        if wseg != seg:
            continue

        # Constant fold: static slot that doesn't match → skip entirely
        if is_static_slot and int(slot_ssa) != wslot:
            continue

        l_skip = st.ctx.next_temp("skip.watch")
        l_fire = st.ctx.next_temp("fire.watch")
        l_join = st.ctx.next_temp("join.watch")

        if is_static_slot:
            # Slot matches statically, only need depth check
            d = "%" + st.ctx.next_temp("depth")
            st.lines.append(f"  {d} = load i32, ptr @bf2.watch.depth, align 4")
            can_fire = "%" + st.ctx.next_temp("can_fire")
            st.lines.append(f"  {can_fire} = icmp slt i32 {d}, 8")
            st.lines.append(f"  br i1 {can_fire}, label %{l_fire}, label %{l_skip}")
        else:
            # Dynamic slot: need both icmp and depth check
            is_slot = "%" + st.ctx.next_temp("is_slot")
            st.lines.append(f"  {is_slot} = icmp eq i32 {slot_ssa}, {wslot}")
            d = "%" + st.ctx.next_temp("depth")
            st.lines.append(f"  {d} = load i32, ptr @bf2.watch.depth, align 4")
            can_fire = "%" + st.ctx.next_temp("can_fire")
            st.lines.append(f"  {can_fire} = icmp slt i32 {d}, 8")
            must_fire = "%" + st.ctx.next_temp("must_fire")
            st.lines.append(f"  {must_fire} = and i1 {is_slot}, {can_fire}")
            st.lines.append(f"  br i1 {must_fire}, label %{l_fire}, label %{l_skip}")

        st.lines.append(f"{l_fire}:")
        nd = "%" + st.ctx.next_temp("ndepth")
        st.lines.append(f"  {nd} = add i32 {d}, 1")
        st.lines.append(f"  store i32 {nd}, ptr @bf2.watch.depth, align 4")
        st.lines.append(f"  call void @bf2.watch.{i}()")
        st.lines.append(f"  store i32 {d}, ptr @bf2.watch.depth, align 4")
        st.lines.append(f"  br label %{l_join}")
        st.lines.append(f"{l_skip}:")
        st.lines.append(f"  br label %{l_join}")
        st.lines.append(f"{l_join}:")


def emit_maybe_watch_current(st: EmitState) -> None:
    """Emit reactor dispatch for the current dynamic cursor position."""
    if not st.watches:
        return

    seg_ptr = "%" + st.ctx.next_temp("cseg_ptr")
    st.lines.append(f"  {seg_ptr} = load ptr, ptr %__cseg, align 8")
    slot_v = "%" + st.ctx.next_temp("cslot_val")
    st.lines.append(f"  {slot_v} = load i32, ptr %__cslot, align 4")

    for i, (wseg, wslot, _) in enumerate(st.watches):
        l_skip = st.ctx.next_temp("skip.watch")
        l_fire = st.ctx.next_temp("fire.watch")
        l_join = st.ctx.next_temp("join.watch")

        wseg_ptr = st.seg_slots[wseg]
        is_seg = "%" + st.ctx.next_temp("is_seg")
        st.lines.append(f"  {is_seg} = icmp eq ptr {seg_ptr}, {wseg_ptr}")
        is_slot = "%" + st.ctx.next_temp("is_slot")
        st.lines.append(f"  {is_slot} = icmp eq i32 {slot_v}, {wslot}")
        match = "%" + st.ctx.next_temp("match")
        st.lines.append(f"  {match} = and i1 {is_seg}, {is_slot}")

        d = "%" + st.ctx.next_temp("depth")
        st.lines.append(f"  {d} = load i32, ptr @bf2.watch.depth, align 4")
        can_fire = "%" + st.ctx.next_temp("can_fire")
        st.lines.append(f"  {can_fire} = icmp slt i32 {d}, 8")

        must_fire = "%" + st.ctx.next_temp("must_fire")
        st.lines.append(f"  {must_fire} = and i1 {match}, {can_fire}")

        st.lines.append(f"  br i1 {must_fire}, label %{l_fire}, label %{l_skip}")
        st.lines.append(f"{l_fire}:")
        nd = "%" + st.ctx.next_temp("ndepth")
        st.lines.append(f"  {nd} = add i32 {d}, 1")
        st.lines.append(f"  store i32 {nd}, ptr @bf2.watch.depth, align 4")
        st.lines.append(f"  call void @bf2.watch.{i}()")
        st.lines.append(f"  store ptr {seg_ptr}, ptr %__cseg, align 8")
        st.lines.append(f"  store i32 {slot_v}, ptr %__cslot, align 4")
        st.lines.append(f"  store i32 {d}, ptr @bf2.watch.depth, align 4")
        st.lines.append(f"  br label %{l_join}")
        st.lines.append(f"{l_skip}:")
        st.lines.append(f"  br label %{l_join}")
        st.lines.append(f"{l_join}:")
