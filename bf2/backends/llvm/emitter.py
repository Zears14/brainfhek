"""LLVM IR emitter — thin orchestrator delegating to sub-modules.

Public API:
    emit_llvm_ir(mod, target) -> ir.Module
"""

from __future__ import annotations

import platform

from llvmlite import ir

from bf2.core import ast as A
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.types import to_ir_type, align, get_struct_type, Int8, Int32, Pointer
from bf2.backends.llvm.emit_preamble import (
    emit_preamble, emit_struct_types, emit_global_segments,
    emit_metadata, emit_string_constants
)
from bf2.backends.llvm.emit_stmts import emit_stmt
from bf2.backends.llvm.emit_watch import try_static_seg_slot, emit_watch_fn
from bf2.core.analysis import find_segment_deps


class LLVMEmitterVisitor:
    """Orchestrates LLVM IR emission by delegating to focused sub-modules."""

    def __init__(self, mod: A.Module, target: str):
        self.st = EmitState(
            mod=mod,
            target=target,
        )

    def emit(self) -> ir.Module:
        st = self.st

        emit_preamble(st)

        for item in st.mod.items:
            if isinstance(item, A.StructDecl):
                st.structs[item.name] = item
            elif isinstance(item, A.FunctionDef):
                st.fns[item.name] = item
            elif isinstance(item, A.SegmentDecl):
                st.global_segs[item.name] = item
            elif isinstance(item, A.ReactorDef):
                sk = try_static_seg_slot(st, item.target)
                if sk:
                    st.watches.append((sk[0], sk[1], item))

        emit_struct_types(st)
        for name, decl in st.structs.items():
            field_types = [f[1] for f in decl.fields]
            get_struct_type(name, field_types, st.module)

        emit_global_segments(st)

        for f in st.fns.values():
            self._emit_function(f)

        for i, (seg, slot, r) in enumerate(st.watches):
            emit_watch_fn(st, i, seg, slot, r)

        emit_metadata(st)
        emit_string_constants(st)

        return st.module

    def _emit_function(self, f: A.FunctionDef) -> None:
        st = self.st
        ret_ty = to_ir_type(f.ret)

        arg_tys = [to_ir_type(t) for _, t in f.params]
        fnty = ir.FunctionType(ret_ty, arg_tys)
        fn = ir.Function(st.module, fnty, name=f.name)
        fn.attributes.add("nounwind")
        entry_block = fn.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)

        st.ctx = LLVMContext(f.ret, builder)
        ctx = st.ctx

        cseg_ptr = ctx.hoist_alloca(Pointer, name="__cseg")
        cslot_ptr = ctx.hoist_alloca(Int32, name="__cslot")
        ctx.locals["__cseg"] = (cseg_ptr, Pointer)
        ctx.locals["__cslot"] = (cslot_ptr, Int32)

        cseg_gv = st.module.globals.get("__bf")
        if cseg_gv is None:
            cseg_gv = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 30000), name="__bf")
            cseg_gv.initializer = ir.Constant(ir.ArrayType(Int8, 30000), None)
            cseg_gv.align = 16

        cseg_typed = builder.bitcast(cseg_gv, Pointer, name="__bf_cast")
        builder.store(cseg_typed, cseg_ptr)
        builder.store(ir.Constant(Int32, 0), cslot_ptr)

        st.ctx.cursor_type = Int8

        for i, (n, t) in enumerate(f.params):
            lty = to_ir_type(t)
            arg = fn.args[i]
            ptr = ctx.hoist_alloca(lty, name=ctx.next_temp(f"p.{n}"))
            builder.store(arg, ptr)
            ctx.locals[n] = (ptr, lty)
            if t.name == "ptr":
                st.ctx.ptr_inner[n] = t.inner or A.TypeRef("i8")
        
        if f.params:
            first_n = f.params[0][0]
            first_p, _ = st.ctx.locals[first_n]
            first_p_cast = builder.bitcast(first_p, Pointer, name=st.ctx.next_temp("cseg_cast"))
            builder.store(first_p_cast, cseg_ptr)
            builder.store(ir.Constant(Int32, 0), cslot_ptr)
            st.ctx.cursor_type = to_ir_type(f.params[0][1])
        
        if f.name == "main":
            for seg in st.one_time_links:
                self._emit_one_time_init(seg, builder)

        for stmt_node in f.body.stmts:
            emit_stmt(st, stmt_node)

        if ret_ty == ir.VoidType():
            if not builder.block.terminator:
                builder.ret_void()
        elif not builder.block.terminator:
            builder.ret(ir.Constant(ret_ty, 0))

    def _emit_one_time_init(self, seg: A.SegmentDecl, builder: ir.IRBuilder) -> None:
        """Emit code to initialize a segment once (standard 'seg' with assignment)."""
        st = self.st
        from bf2.backends.llvm.emit_expr import emit_expr
        from bf2.backends.llvm.emit_expr import _coerce
        from bf2.backends.llvm.types import Int32
        
        loop_cond = builder.append_basic_block(name=st.ctx.next_temp("init.cond"))
        loop_body = builder.append_basic_block(name=st.ctx.next_temp("init.body"))
        loop_end = builder.append_basic_block(name=st.ctx.next_temp("init.end"))
        
        i_ptr = builder.alloca(Int32, name=st.ctx.next_temp("init.i"))
        builder.store(ir.Constant(Int32, 0), i_ptr)
        builder.branch(loop_cond)
        
        builder.position_at_end(loop_cond)
        i = builder.load(i_ptr)
        cond = builder.icmp_signed("<", i, ir.Constant(Int32, seg.length))
        builder.cbranch(cond, loop_body, loop_end)
        
        builder.position_at_end(loop_body)
        from bf2.backends.llvm.emit_reactive import _bind_expr_to_index
        bound_init = _bind_expr_to_index(seg.init, i)
        
        val, ty = emit_expr(st, bound_init, st.ctx)
        lty = to_ir_type(seg.elem_type)
        val = _coerce(st, val, ty, lty, st.ctx)
        
        target_ptr = st.seg_slots[seg.name]
        dest_ptr = builder.gep(target_ptr, [ir.Constant(Int32, 0), i], inbounds=True)
        builder.store(val, dest_ptr, align=align(lty))
        
        ni = builder.add(i, ir.Constant(Int32, 1))
        builder.store(ni, i_ptr)
        builder.branch(loop_cond)
        
        builder.position_at_end(loop_end)


def emit_llvm_ir(mod: A.Module, target: str | None = None) -> ir.Module:
    """Public API: emit LLVM IR for a BF2 module (returns ir.Module)."""
    from bf2.backends.llvm.types import clear_type_caches
    clear_type_caches()
    if target is None:
        target = f"{platform.machine()}-pc-linux-gnu"
    return LLVMEmitterVisitor(mod, target).emit()


def lty_align(lty: str) -> int:
    """Alignment helper using the types module."""
    return align(lty)
