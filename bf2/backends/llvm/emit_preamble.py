"""Emit the LLVM IR preamble: target triple, extern declares, globals."""

from __future__ import annotations

from llvmlite import ir

from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.types import (
    to_ir_type, align, LINUX_LIBC,
    Int8, Int32, Int64, Pointer
)


def emit_preamble(st: EmitState) -> None:
    """Emit target triple, external declarations, built-in globals and format strings."""
    st.module = ir.Module(name="bf2_module", context=ir.Context())
    st.module.triple = st.target

    use_linux = getattr(st.mod, "use_linux_stdlib", False)

    if not use_linux:
        declare_extern(st, "putchar", ir.FunctionType(Int32, [Int32]))
        declare_extern(st, "getchar", ir.FunctionType(Int32, []))
        declare_extern(st, "scanf", ir.FunctionType(Int32, [Pointer], var_arg=True))
        declare_extern(st, "printf", ir.FunctionType(Int32, [Pointer], var_arg=True))
        declare_extern(st, "strlen", ir.FunctionType(Int64, [Pointer]))
        declare_extern(st, "malloc", ir.FunctionType(Pointer, [Int64]))
        declare_extern(st, "free", ir.FunctionType(ir.VoidType(), [Pointer]))

    if use_linux:
        for name, (ret_str, args) in LINUX_LIBC.items():
            ret_ty = get_linux_type(ret_str)
            var_arg = "..." in args
            arg_tys = [get_linux_type(a) for a in args if a != "..."]
            fnty = ir.FunctionType(ret_ty, arg_tys, var_arg=var_arg)
            declare_extern(st, name, fnty)

    tape_ty = ir.ArrayType(Int8, 30000)
    tape = ir.GlobalVariable(st.module, tape_ty, name="__bf")
    tape.initializer = ir.Constant(tape_ty, None)
    tape.align = 16


    fmt_i = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 4), name="__bf2_fmt_i")
    fmt_i.initializer = ir.Constant(ir.ArrayType(Int8, 4), bytearray(b"%d\n\0"))
    fmt_i.align = 1
    fmt_i.linkage = "private"
    fmt_i.unnamed_addr = True

    fmt_ir = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 3), name="__bf2_fmt_ir")
    fmt_ir.initializer = ir.Constant(ir.ArrayType(Int8, 3), bytearray(b"%d\0"))
    fmt_ir.align = 1
    fmt_ir.linkage = "private"
    fmt_ir.unnamed_addr = True

    fmt_i64 = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 5), name="__bf2_fmt_i64")
    fmt_i64.initializer = ir.Constant(ir.ArrayType(Int8, 5), bytearray(b"%ld\n\0"))
    fmt_i64.align = 1
    fmt_i64.linkage = "private"
    fmt_i64.unnamed_addr = True

    fmt_i64r = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 4), name="__bf2_fmt_i64r")
    fmt_i64r.initializer = ir.Constant(ir.ArrayType(Int8, 4), bytearray(b"%ld\0"))
    fmt_i64r.align = 1
    fmt_i64r.linkage = "private"
    fmt_i64r.unnamed_addr = True

    fmt_f = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 7), name="__bf2_fmt_f")
    fmt_f.initializer = ir.Constant(ir.ArrayType(Int8, 7), bytearray(b"%.17g\n\0"))
    fmt_f.align = 1
    fmt_f.linkage = "private"
    fmt_f.unnamed_addr = True

    fmt_fr = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 6), name="__bf2_fmt_fr")
    fmt_fr.initializer = ir.Constant(ir.ArrayType(Int8, 6), bytearray(b"%.17g\0"))
    fmt_fr.align = 1
    fmt_fr.linkage = "private"
    fmt_fr.unnamed_addr = True

    fmt_s = ir.GlobalVariable(st.module, ir.ArrayType(Int8, 4), name="__bf2_fmt_s")
    fmt_s.initializer = ir.Constant(ir.ArrayType(Int8, 4), bytearray(b"%s\0\0"))
    fmt_s.align = 1
    fmt_s.linkage = "private"
    fmt_s.unnamed_addr = True

    watch_depth = ir.GlobalVariable(st.module, Int32, name="bf2.watch.depth")
    watch_depth.initializer = ir.Constant(Int32, 0)
    watch_depth.align = 4

    watch_mask = ir.GlobalVariable(st.module, ir.IntType(64), name="bf2.watch.mask")
    watch_mask.initializer = ir.Constant(ir.IntType(64), 0)
    watch_mask.align = 8

    if use_linux:
        pass  # timespec type handling removed - use by-name reference

        stdout = ir.GlobalVariable(st.module, Int32, name="STDOUT")
        stdout.initializer = ir.Constant(Int32, 1)
        stdout.align = 4

        o_creat = ir.GlobalVariable(st.module, Int32, name="O_CREAT")
        o_creat.initializer = ir.Constant(Int32, 64)
        o_creat.align = 4

        declare_extern(st, "snprintf", ir.FunctionType(Int32, [Pointer, Int64, Pointer], var_arg=True))


def declare_extern(st: EmitState, name: str, fnty: ir.FunctionType) -> None:
    """Declare an external function if not already declared."""
    if name in st.declared_externs:
        return
    st.declared_externs.add(name)
    ir.Function(st.module, fnty, name=name)


def get_linux_type(name: str) -> ir.Type:
    """Map Linux libc type name to llvmlite type."""
    if name == "i32":
        return Int32
    if name == "i64":
        return Int64
    if name == "ptr":
        return Pointer
    if name == "void":
        return ir.VoidType()
    return Int32


def emit_struct_types(st: EmitState) -> None:
    """Emit struct types for all collected struct declarations."""
    pass  # Struct types now use LiteralStructType


def emit_global_segments(st: EmitState) -> None:
    """Emit global segments for all collected segment declarations."""
    from bf2.core.analysis import find_segment_deps
    for name, seg in st.global_segs.items():
        lty = to_ir_type(seg.elem_type)
        arr_ty = ir.ArrayType(lty, seg.length)
        gv = ir.GlobalVariable(st.module, arr_ty, name=name)
        gv.initializer = ir.Constant(arr_ty, None)
        gv.align = align(lty)
        st.seg_slots[name] = gv

        if seg.init:
            if seg.reactive:
                # Create a global activation flag for this link
                active_gv = ir.GlobalVariable(st.module, ir.IntType(1), name=f"bf2.link.active.{name}")
                active_gv.initializer = ir.Constant(ir.IntType(1), 1)
                st.link_active_gv[name] = active_gv

                deps = find_segment_deps(seg.init)
                for d in deps:
                    if d not in st.links:
                        st.links[d] = []
                    st.links[d].append(seg)
            else:
                st.one_time_links.append(seg)




def emit_metadata(st: EmitState) -> None:
    """Emit collected metadata nodes at the end of the module."""
    for md in st.loop_metadata:
        pass


def emit_string_constants(st: EmitState) -> None:
    """Emit global string constants for literals at the end of the module."""
    for val, name in st.string_constants.items():
        if st.module.globals.get(name) is not None:
            continue
        const_arr = ir.ArrayType(Int8, len(val) + 1)
        byte_arr = bytearray(val.encode("utf-8")) + b"\0"
        gv = ir.GlobalVariable(st.module, const_arr, name=name)
        gv.initializer = ir.Constant(const_arr, byte_arr)
        gv.align = 1
        gv.linkage = "private"
        gv.unnamed_addr = True
