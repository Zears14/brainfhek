"""LLVM IR emitter — thin orchestrator delegating to sub-modules.

Public API:
    emit_llvm_ir(mod, target) -> str
"""

from __future__ import annotations

import platform

from bf2.core import ast as A
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.types import scalar_ty
from bf2.backends.llvm.emit_preamble import (
    emit_preamble, emit_struct_types, emit_global_segments,
    emit_fn_attrs, emit_metadata, emit_string_constants
)
from bf2.backends.llvm.emit_stmts import emit_stmt
from bf2.backends.llvm.emit_watch import try_static_seg_slot, emit_watch_fn


class LLVMEmitterVisitor:
    """Orchestrates LLVM IR emission by delegating to focused sub-modules."""

    def __init__(self, mod: A.Module, target: str):
        self.st = EmitState(
            mod=mod,
            target=target,
            ctx=LLVMContext("void"),
        )
        # Register attribute group #0 = nounwind
        self.st.fn_attrs["#0"] = "nounwind"

    def emit(self) -> str:
        st = self.st

        # 1. Preamble
        emit_preamble(st)

        # 2. Collect top-level declarations
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

        # 3. Struct types
        emit_struct_types(st)

        # 4. Global segments
        emit_global_segments(st)

        # 5. Functions
        for f in st.fns.values():
            self._emit_function(f)

        # 6. Reactors
        for i, (seg, slot, r) in enumerate(st.watches):
            emit_watch_fn(st, i, seg, slot, r)

        # 7. Attribute groups, metadata and string constants (appended at module end)
        emit_fn_attrs(st)
        emit_metadata(st)
        emit_string_constants(st)

        return "\n".join(st.lines)

    def _emit_function(self, f: A.FunctionDef) -> None:
        st = self.st
        ret_ty = scalar_ty(f.ret)
        st.ctx = LLVMContext(ret_ty)
        st.alloca_lines = []
        args = ", ".join(f"{scalar_ty(t)} %{n}" for n, t in f.params)
        st.lines.append(f"define {ret_ty} @{f.name}({args}) #0 {{")
        st.lines.append("entry:")
        insertion_point = len(st.lines)

        # Local cursor allocas (promotable by mem2reg)
        st.alloca_lines.append("  %__cseg = alloca ptr, align 8")
        st.alloca_lines.append("  %__cslot = alloca i32, align 4")

        # Alloc parameters
        for n, t in f.params:
            lty = scalar_ty(t)
            p = "%" + st.ctx.next_temp(f"p.{n}")
            st.alloca_lines.append(f"  {p} = alloca {lty}, align {lty_align(lty)}")
            st.lines.append(f"  store {lty} %{n}, ptr {p}, align {lty_align(lty)}")
            st.ctx.locals[n] = (p, lty)
            if t.name == "ptr":
                st.ctx.ptr_inner[n] = t.inner or A.TypeRef("i8")

        if f.params:
            first_n = f.params[0][0]
            first_p, _ = st.ctx.locals[first_n]
            st.lines.append(f"  store ptr {first_p}, ptr %__cseg, align 8")
            st.lines.append("  store i32 0, ptr %__cslot, align 4")
            st.ctx.cursor_type = scalar_ty(f.params[0][1])
        else:
            st.lines.append("  store ptr @__bf, ptr %__cseg, align 8")
            st.lines.append("  store i32 0, ptr %__cslot, align 4")
            st.ctx.cursor_type = "i8"

        for stmt_node in f.body.stmts:
            emit_stmt(st, stmt_node)

        # Hoist allocas to entry block
        st.lines[insertion_point:insertion_point] = st.alloca_lines

        if ret_ty == "void":
            st.lines.append("  ret void")
        elif not st.lines[-1].strip().startswith("ret "):
            st.lines.append(f"  ret {ret_ty} 0")

        st.lines.append("}")


def lty_align(lty: str) -> int:
    """Alignment helper using the types module."""
    from bf2.backends.llvm.types import align
    return align(lty)


def emit_llvm_ir(mod: A.Module, target: str | None = None) -> str:
    """Public API: emit LLVM IR for a BF2 module."""
    if target is None:
        target = f"{platform.machine()}-pc-linux-gnu"
    return LLVMEmitterVisitor(mod, target).emit()
