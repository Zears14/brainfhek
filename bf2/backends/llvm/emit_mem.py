"""Memory operations: GEP resolution, cursor access, alloc/free, pointer ops."""

from __future__ import annotations

from typing import Any, Tuple

from bf2.core import ast as A
from bf2.core.errors import BF2Error
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.types import scalar_ty, align


class LLVMGenError(BF2Error):
    """Raised when LLVM IR generation fails."""


def get_current_cell_ptr(st: EmitState) -> Tuple[str, str]:
    """Load the current cursor (seg pointer + slot) and return (gep_ptr, llvm_type)."""
    bp = "%" + st.ctx.next_temp("cseg")
    st.lines.append(f"  {bp} = load ptr, ptr %__cseg, align 8")
    idx = "%" + st.ctx.next_temp("cslot")
    st.lines.append(f"  {idx} = load i32, ptr %__cslot, align 4")
    lty = st.ctx.cursor_type
    p = "%" + st.ctx.next_temp("cptr")
    st.lines.append(f"  {p} = getelementptr {lty}, ptr {bp}, i32 {idx}")
    return (p, lty)


def gep(st: EmitState, r: A.RefExpr, ctx: LLVMContext) -> Tuple[str, str]:
    """Resolve a RefExpr to an (LLVM pointer SSA value, element type) tuple."""
    if r.parts and r.parts[0] == "@":
        base_n = r.parts[1]
        p = st.seg_slots.get(base_n)
        seg = st.global_segs.get(base_n)
        lty = scalar_ty(seg.elem_type)
        return (p, lty)

    head = str(r.parts[0])

    if head in ctx.locals:
        p, ty = ctx.locals[head]
        lty = ty
        slot = 1
        while slot < len(r.parts):
            sub = r.parts[slot]
            if isinstance(sub, A.Expr):
                from bf2.backends.llvm.emit_expr import emit_expr
                idx_v, _ = emit_expr(st, sub, ctx)
                if lty == "ptr":
                    pb = "%" + ctx.next_temp("pb")
                    st.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
                    p = "%" + ctx.next_temp("ptr")
                    inn = ctx.ptr_inner.get(head, A.TypeRef("i8"))
                    lty = scalar_ty(inn)
                    st.lines.append(f"  {p} = getelementptr {lty}, ptr {pb}, i32 {idx_v}")
                elif lty.startswith("["):
                    inn_ty = lty[lty.find("x") + 1 : -1].strip()
                    p_old = p
                    p = "%" + ctx.next_temp("ptr")
                    st.lines.append(f"  {p} = getelementptr {lty}, ptr {p_old}, i32 0, i32 {idx_v}")
                    lty = inn_ty
            elif isinstance(sub, str):
                # Field access
                s_name = ""
                if lty.startswith("%struct."):
                    s_name = lty[8:]
                elif lty == "ptr":
                    inn = st.ctx.ptr_inner.get(head, A.TypeRef("i8"))
                    s_name = inn.name

                if not s_name or s_name not in st.structs:
                    raise LLVMGenError(f"Cannot resolve struct field {sub} on type {lty}")

                s_decl = st.structs[s_name]
                field_idx = -1
                for i, (fn, ft) in enumerate(s_decl.fields):
                    if fn == sub:
                        field_idx = i
                        lty = scalar_ty(ft)
                        if ft.name == "ptr":
                            st.ctx.ptr_inner[f"__tmp_{s_name}_{sub}"] = ft.inner or A.TypeRef("i8")
                            head = f"__tmp_{s_name}_{sub}"
                        break
                if field_idx == -1:
                    raise LLVMGenError(f"Field {sub} not found in struct {s_name}")
                p_old = p
                p = f"%{st.ctx.next_temp('ptr')}"
                st.lines.append(f"  {p} = getelementptr %struct.{s_name}, ptr {p_old}, i32 0, i32 {field_idx}")
            slot += 1
        return (p, lty)

    if head in st.seg_slots:
        p = st.seg_slots[head]
        seg = st.global_segs[head]
        lty = scalar_ty(seg.elem_type)
        p_res = p
        full_ty = f"[{seg.length} x {lty}]"
        slot = 1
        while slot < len(r.parts):
            sub = r.parts[slot]
            if isinstance(sub, A.Expr):
                from bf2.backends.llvm.emit_expr import emit_expr
                idx_v, _ = emit_expr(st, sub, ctx)
                p_old = p_res
                p_res = "%" + ctx.next_temp("ptr")
                st.lines.append(f"  {p_res} = getelementptr {full_ty}, ptr {p_old}, i32 0, i32 {idx_v}")
                full_ty = lty
            slot += 1
        return (p_res, lty)

    raise LLVMGenError(f"Cannot resolve reference {r.parts}")


def emit_alloc(st: EmitState, s: A.AllocStmt) -> None:
    """Emit a malloc call and optional local pointer binding."""
    ity = scalar_ty(s.ty)
    isz = align(ity)
    total = s.count * isz
    r = "%" + st.ctx.next_temp("mem")
    st.lines.append(f"  {r} = call ptr @malloc(i64 {total})")
    if s.name:
        p = "%" + st.ctx.next_temp(f"v.{s.name}")
        st.alloca_lines.append(f"  {p} = alloca ptr, align 8")
        st.lines.append(f"  store ptr {r}, ptr {p}, align 8")
        st.ctx.locals[s.name] = (p, "ptr")
        st.ctx.ptr_inner[s.name] = s.ty


def emit_free(st: EmitState, ptr_name: str) -> None:
    """Emit a free call for the named pointer."""
    p, _ = st.ctx.locals[ptr_name]
    v = "%" + st.ctx.next_temp("v")
    st.lines.append(f"  {v} = load ptr, ptr {p}, align 8")
    st.lines.append(f"  call void @free(ptr {v})")


def emit_ptr_arith(st: EmitState, s: A.PtrArith) -> None:
    """Emit pointer arithmetic (ptr += delta)."""
    p, ty = st.ctx.locals[s.name]
    v = "%" + st.ctx.next_temp("ptr")
    st.lines.append(f"  {v} = load ptr, ptr {p}, align 8")
    nv = "%" + st.ctx.next_temp("nptr")
    st.lines.append(f"  {nv} = getelementptr i8, ptr {v}, i32 {s.delta}")
    st.lines.append(f"  store ptr {nv}, ptr {p}, align 8")


def emit_ptr_read(st: EmitState, ptr_name: str, loc: Any) -> None:
    """Dereference a pointer and store the result into the current cell."""
    p, _ = st.ctx.locals[ptr_name]
    inn = st.ctx.ptr_inner.get(ptr_name, A.TypeRef("i8"))
    pb = "%" + st.ctx.next_temp("pb")
    st.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
    ity = scalar_ty(inn)
    v = "%" + st.ctx.next_temp("v")
    st.lines.append(f"  {v} = load {ity}, ptr {pb}, align {align(ity)}")
    cp, cty = get_current_cell_ptr(st)
    tv = "%" + st.ctx.next_temp("conv")
    if cty == ity:
        tv = v
    else:
        st.lines.append(f"  {tv} = sext {ity} {v} to {cty}")
    st.lines.append(f"  store {cty} {tv}, ptr {cp}, align {align(cty)}")


def emit_ptr_write(st: EmitState, s: A.PtrWrite) -> None:
    """Write an expression value through a pointer."""
    from bf2.backends.llvm.emit_expr import emit_expr
    p, _ = st.ctx.locals[s.ptr]
    inn = st.ctx.ptr_inner.get(s.ptr, A.TypeRef("i8"))
    pb = "%" + st.ctx.next_temp("pb")
    st.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
    rv, rt = emit_expr(st, s.value, st.ctx)
    ity = scalar_ty(inn)
    st.lines.append(f"  store {ity} {rv}, ptr {pb}, align {align(ity)}")
