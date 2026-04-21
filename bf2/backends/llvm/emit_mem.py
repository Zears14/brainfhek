"""Memory operations: GEP resolution, cursor access, alloc/free, pointer ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from llvmlite import ir

from bf2.core import ast as A
from bf2.core.errors import BF2Error
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.types import to_ir_type, align, Int32, Int64, Pointer

if TYPE_CHECKING:
    from bf2.backends.llvm.types import ir


class LLVMGenError(BF2Error):
    """Raised when LLVM IR generation fails."""


def get_current_cell_ptr(st: EmitState) -> Tuple[ir.Value, ir.Type]:
    """Load the current cursor (seg pointer + slot) and return (gep_ptr, llvm_type)."""
    ctx = st.ctx
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
    cseg_val = ctx.builder.load(cseg_ptr, name=ctx.next_temp("cseg"))
    cslot = ctx.builder.load(cslot_ptr, name=ctx.next_temp("cslot"))
    lty = ctx.cursor_type
    typed_seg = cseg_val
    typed_ptr_ty = ir.PointerType(lty)
    if typed_seg.type != typed_ptr_ty:
        typed_seg = ctx.builder.bitcast(typed_seg, typed_ptr_ty, name=ctx.next_temp("cseg.cast"))
    ptr = ctx.builder.gep(typed_seg, [cslot], inbounds=True, name=ctx.next_temp("cptr"))
    return (ptr, lty)


def gep(st: EmitState, r: A.RefExpr, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    """Resolve a RefExpr to an (ir.Value pointer, ir.Type element type) tuple."""
    has_at = r.parts and r.parts[0] == "@"
    base_idx = 1 if has_at else 0
    base_n = r.parts[base_idx]

    gv = st.global_segs.get(base_n)
    if gv is not None:
        base_ptr = st.seg_slots[base_n]
        lty = to_ir_type(gv.elem_type)

        has_field = len(r.parts) > base_idx + 1 and gv.elem_type.name in st.structs
        if has_field:
            s_name = gv.elem_type.name
            field_name = r.parts[base_idx + 1]
            s_decl = st.structs[s_name]

            field_idx = -1
            for i, (fn, ft) in enumerate(s_decl.fields):
                if fn == field_name:
                    field_idx = i
                    lty = to_ir_type(ft)
                    break

            if field_idx == -1:
                raise LLVMGenError(f"Field {field_name} not found in struct {s_name}")

            field_ptr = ctx.builder.gep(base_ptr, [ir.Constant(Int32, 0), ir.Constant(Int32, 0), ir.Constant(Int32, field_idx)], inbounds=True, name=ctx.next_temp("fptr"))
            return (field_ptr, lty)

        if len(r.parts) > base_idx + 1:
            sub = r.parts[base_idx + 1]
            if isinstance(sub, int):
                idx = sub
            elif isinstance(sub, A.IntLit):
                idx = sub.value
            else:
                from bf2.backends.llvm.emit_expr import emit_expr
                idx_v, _ = emit_expr(st, sub, ctx)
                arr_ptr = ctx.builder.gep(base_ptr, [ir.Constant(Int32, 0), idx_v], inbounds=True, name=ctx.next_temp("arr"))
                return (arr_ptr, lty)
            arr_ptr = ctx.builder.gep(base_ptr, [ir.Constant(Int32, 0), ir.Constant(Int32, idx)], inbounds=True, name=ctx.next_temp("arr"))
            return (arr_ptr, lty)

        return (base_ptr, lty)

    head = str(r.parts[0])

    if head in ctx.locals:
        ptr, ty = ctx.locals[head]
        lty = ty
        slot = 1
        while slot < len(r.parts):
            sub = r.parts[slot]
            if isinstance(sub, A.Expr):
                from bf2.backends.llvm.emit_expr import emit_expr
                idx_v, _ = emit_expr(st, sub, ctx)
                if isinstance(lty, ir.PointerType):
                    pb = ctx.builder.load(ptr, align=8, name=ctx.next_temp("pb"))
                    inn = ctx.ptr_inner.get(head, A.TypeRef("i8"))
                    lty = to_ir_type(inn)
                    typed_ptr_ty = ir.PointerType(lty)
                    if pb.type != typed_ptr_ty:
                        pb = ctx.builder.bitcast(pb, typed_ptr_ty, name=ctx.next_temp("pbc"))
                    ptr = ctx.builder.gep(pb, [idx_v], inbounds=True, name=ctx.next_temp("ptr"))
                elif isinstance(lty, ir.ArrayType):
                    inn_ty = lty.element
                    ptr_old = ptr
                    ptr = ctx.builder.gep(ptr_old, [ir.Constant(Int32, 0), idx_v], inbounds=True, name=ctx.next_temp("ptr"))
                    lty = inn_ty
            elif isinstance(sub, str):
                # For struct field access, determine struct name from the RefExpr context
                # Look at the base variable to find its type
                s_name = None
                container_ty = lty
                
                # Try to find struct name from the head of the RefExpr
                base_name = str(r.parts[0])
                
                # Check if it's a global segment
                if base_name in st.global_segs:
                    seg = st.global_segs[base_name]
                    # If it's a struct type, get the struct name
                    if hasattr(seg, 'elem_type'):
                        # This is a segment - get element type
                        if seg.elem_type.name in st.structs:
                            s_name = seg.elem_type.name
                # Check local variables
                elif base_name in ctx.locals:
                    ptr_var, _ = ctx.locals[base_name]
                    # Check if we have ptr_inner info
                    if base_name in ctx.ptr_inner:
                        inner_type = ctx.ptr_inner[base_name]
                        if inner_type.name in st.structs:
                            s_name = inner_type.name
                
                if not s_name or s_name not in st.structs:
                    raise LLVMGenError(f"Cannot resolve struct field {sub} on reference {r.parts}")

                s_decl = st.structs[s_name]
                field_idx = -1
                for i, (fn, ft) in enumerate(s_decl.fields):
                    if fn == sub:
                        field_idx = i
                        lty = to_ir_type(ft)
                        if ft.name == "ptr":
                            ctx.ptr_inner[f"__tmp_{s_name}_{sub}"] = ft.inner or A.TypeRef("i8")
                            head = f"__tmp_{s_name}_{sub}"
                        break
                if field_idx == -1:
                    raise LLVMGenError(f"Field {sub} not found in struct {s_name}")
                ptr_old = ptr
                if isinstance(container_ty, ir.PointerType):
                    ptr_old = ctx.builder.load(ptr_old, align=8, name=ctx.next_temp("sptr"))
                ptr = ctx.builder.gep(ptr_old, [ir.Constant(Int32, 0), ir.Constant(Int32, field_idx)], inbounds=True, name=ctx.next_temp("ptr"))
            slot += 1
        return (ptr, lty)

    if head in st.seg_slots:
        gv = st.seg_slots[head]
        seg = st.global_segs[head]
        lty = to_ir_type(seg.elem_type)
        p_res = gv
        slot = 1
        while slot < len(r.parts):
            sub = r.parts[slot]
            if isinstance(sub, A.Expr):
                from bf2.backends.llvm.emit_expr import emit_expr
                idx_v, _ = emit_expr(st, sub, ctx)
                p_old = p_res
                p_res = ctx.builder.gep(p_old, [ir.Constant(Int32, 0), idx_v], inbounds=True, name=ctx.next_temp("ptr"))
            slot += 1
        return (p_res, lty)

    raise LLVMGenError(f"Cannot resolve reference {r.parts}")


def emit_alloc(st: EmitState, s: A.AllocStmt) -> None:
    """Emit a malloc call and optional local pointer binding."""
    ctx = st.ctx
    ity = to_ir_type(s.ty)
    ptr_ty = to_ir_type(A.TypeRef("ptr", s.ty))
    isz = align(ity)
    total = s.count * isz
    malloc_fn = st.module.globals.get("malloc")
    if malloc_fn is None:
        malloc_ty = ir.FunctionType(Pointer, [Int64])
        malloc_fn = ir.Function(st.module, malloc_ty, name="malloc")
    r = ctx.builder.call(malloc_fn, [ir.Constant(Int64, total)], name=ctx.next_temp("mem"))
    if r.type != ptr_ty:
        r = ctx.builder.bitcast(r, ptr_ty, name=ctx.next_temp("memcast"))
    if s.name:
        ptr = ctx.hoist_alloca(ptr_ty, name=ctx.next_temp(f"v.{s.name}"))
        ctx.builder.store(r, ptr, align=8)
        ctx.locals[s.name] = (ptr, ptr_ty)
        ctx.ptr_inner[s.name] = s.ty


def emit_free(st: EmitState, ptr_name: str) -> None:
    """Emit a free call for the named pointer."""
    ctx = st.ctx
    ptr, _ = ctx.locals[ptr_name]
    v = ctx.builder.load(ptr, align=8, name=ctx.next_temp("v"))
    if v.type != Pointer:
        v = ctx.builder.bitcast(v, Pointer, name=ctx.next_temp("freecast"))
    free_fn = st.module.globals.get("free")
    if free_fn is None:
        free_ty = ir.FunctionType(ir.VoidType(), [Pointer])
        free_fn = ir.Function(st.module, free_ty, name="free")
    ctx.builder.call(free_fn, [v])


def emit_ptr_arith(st: EmitState, s: A.PtrArith) -> None:
    """Emit pointer arithmetic (ptr += delta)."""
    ctx = st.ctx
    ptr, _ = ctx.locals[s.name]
    v = ctx.builder.load(ptr, align=8, name=ctx.next_temp("ptr"))
    nv = ctx.builder.gep(v, [ir.Constant(Int32, s.delta)], inbounds=True, name=ctx.next_temp("nptr"))
    ctx.builder.store(nv, ptr, align=8)


def emit_ptr_read(st: EmitState, ptr_name: str, loc: Any) -> None:
    """Dereference a pointer and store the result into the current cell."""
    ctx = st.ctx
    ptr, _ = ctx.locals[ptr_name]
    inn = ctx.ptr_inner.get(ptr_name, A.TypeRef("i8"))
    pb = ctx.builder.load(ptr, align=8, name=ctx.next_temp("pb"))
    ity = to_ir_type(inn)
    if pb.type != ir.PointerType(ity):
        pb = ctx.builder.bitcast(pb, ir.PointerType(ity), name=ctx.next_temp("pbc"))
    v = ctx.builder.load(pb, align=align(ity), name=ctx.next_temp("v"))
    cp, cty = get_current_cell_ptr(st)
    if cty == ity:
        tv = v
    else:
        tv = ctx.builder.sext(v, cty, name=ctx.next_temp("conv"))
    ctx.builder.store(tv, cp, align=align(cty))


def emit_ptr_write(st: EmitState, s: A.PtrWrite) -> None:
    """Write an expression value through a pointer."""
    from bf2.backends.llvm.emit_expr import emit_expr
    ctx = st.ctx
    ptr, _ = ctx.locals[s.ptr]
    inn = ctx.ptr_inner.get(s.ptr, A.TypeRef("i8"))
    pb = ctx.builder.load(ptr, align=8, name=ctx.next_temp("pb"))
    rv, rt = emit_expr(st, s.value, st.ctx)
    ity = to_ir_type(inn)
    if pb.type != ir.PointerType(ity):
        pb = ctx.builder.bitcast(pb, ir.PointerType(ity), name=ctx.next_temp("pbc"))
    if rv.type != ity:
        from bf2.backends.llvm.emit_expr import _coerce
        rv = _coerce(st, rv, rt, ity, ctx)
    ctx.builder.store(rv, pb, align=align(ity))
