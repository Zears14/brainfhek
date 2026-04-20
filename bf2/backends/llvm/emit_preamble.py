"""Emit the LLVM IR preamble: target triple, extern declares, globals."""

from __future__ import annotations

from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.types import scalar_ty, align, LINUX_LIBC


def emit_preamble(st: EmitState) -> None:
    """Emit target triple, external declarations, built-in globals and format strings."""
    st.lines.append(f'target triple = "{st.target}"')

    use_linux = getattr(st.mod, "use_linux_stdlib", False)
    if not use_linux:
        st.lines.append("declare i32 @putchar(i32)")
        st.lines.append("declare i32 @getchar()")
        st.lines.append("declare i32 @scanf(ptr, ...)")
        st.lines.append("declare i32 @printf(ptr, ...)")
        st.lines.append("declare i64 @strlen(ptr)")
        st.lines.append("declare ptr @malloc(i64)")
        st.lines.append("declare void @free(ptr)")

    if use_linux:
        for name, (ret, args) in LINUX_LIBC.items():
            if name == "printf":
                continue
            arg_s = ", ".join(args)
            st.lines.append(f"declare {ret} @{name}({arg_s})")

    # Built-in BF tape
    st.lines.append("@__bf = global [30000 x i8] zeroinitializer, align 16")

    # Format strings for IO
    st.lines.append('@__.fmt_i = private unnamed_addr constant [4 x i8] c"%d\\0A\\00", align 1')
    st.lines.append('@__.fmt_ir = private unnamed_addr constant [3 x i8] c"%d\\00", align 1')
    st.lines.append('@__.fmt_i64 = private unnamed_addr constant [5 x i8] c"%ld\\0A\\00", align 1')
    st.lines.append('@__.fmt_i64r = private unnamed_addr constant [4 x i8] c"%ld\\00", align 1')
    st.lines.append('@__.fmt_f = private unnamed_addr constant [7 x i8] c"%.17g\\0A\\00", align 1')
    st.lines.append('@__.fmt_fr = private unnamed_addr constant [6 x i8] c"%.17g\\00", align 1')
    st.lines.append('@__.fmt_s = private unnamed_addr constant [4 x i8] c"%s\\00\\00", align 1')

    # Reactor depth counter
    st.lines.append("@bf2.watch.depth = global i32 0, align 4")

    if use_linux:
        st.lines.append("%struct.timespec = type { i64, i64 }")
        st.lines.append("@STDOUT = global i32 1, align 4")
        st.lines.append("@O_CREAT = global i32 64, align 4")
        st.lines.append("declare i32 @snprintf(ptr, i64, ptr, ...)")


def emit_struct_types(st: EmitState) -> None:
    """Emit ``%struct.X = type { ... }`` for all collected struct declarations."""
    for s in st.structs.values():
        fields = ", ".join(scalar_ty(t) for _, t in s.fields)
        st.lines.append(f"%struct.{s.name} = type {{ {fields} }}")


def emit_global_segments(st: EmitState) -> None:
    """Emit ``@name = global [N x T] zeroinitializer`` for all global segments."""
    for s in st.global_segs.values():
        lty = scalar_ty(s.elem_type)
        sz = s.length
        st.lines.append(f"@{s.name} = global [{sz} x {lty}] zeroinitializer, align {align(lty)}")
        st.seg_slots[s.name] = f"@{s.name}"


def emit_fn_attrs(st: EmitState) -> None:
    """Emit function attribute groups at the end of the module."""
    if st.fn_attrs:
        for group_id, attrs in st.fn_attrs.items():
            st.lines.append(f"attributes {group_id} = {{ {attrs} }}")


def emit_metadata(st: EmitState) -> None:
    """Emit collected metadata nodes at the end of the module."""
    for md in st.loop_metadata:
        st.lines.append(md)


def emit_string_constants(st: EmitState) -> None:
    """Emit global string constants for literals at the end of the module."""
    for val, name in st.string_constants.items():
        # Escape any control characters for LLVM IR string syntax
        # e.g. \n -> \0A, " -> \22
        def b_esc(c: str) -> str:
            o = ord(c)
            if o < 32 or o >= 127 or c in ('"', '\\'):
                return f"\\{o:02X}"
            return c
        esc_val = "".join(b_esc(c) for c in val)
        st.lines.append(f'{name} = private unnamed_addr constant [{len(val) + 1} x i8] c"{esc_val}\\00", align 1')
