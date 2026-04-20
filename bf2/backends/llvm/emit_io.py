"""IO statement codegen: `.`, `.i`, `.f`, `.s`, `,` handlers."""

from __future__ import annotations

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.emit_mem import get_current_cell_ptr
from bf2.backends.llvm.emit_expr import emit_expr
from bf2.backends.llvm.types import align


def emit_io(st: EmitState, s: A.IOStmt) -> None:
    """Emit IO operations for all `.` and `,` variants."""
    use_linux = getattr(st.mod, "use_linux_stdlib", False)

    if s.kind == ".":
        _emit_putchar(st, use_linux)
    elif s.kind == ".i" and s.expr:
        _emit_print_expr(st, s.expr, "@__.fmt_i", use_linux)
    elif s.kind == ".ir" and s.expr:
        _emit_print_expr(st, s.expr, "@__.fmt_ir", use_linux)
    elif s.kind == ".i64" and s.expr:
        _emit_print_expr(st, s.expr, "@__.fmt_i64", use_linux)
    elif s.kind == ".i64r" and s.expr:
        _emit_print_expr(st, s.expr, "@__.fmt_i64r", use_linux)
    elif (s.kind == ".f" or s.kind == ".fr") and s.expr:
        fmt = "@__.fmt_f" if s.kind == ".f" else "@__.fmt_fr"
        _emit_print_expr(st, s.expr, fmt, use_linux)
    elif s.kind == ".s" and s.expr:
        _emit_print_expr(st, s.expr, "@__.fmt_s", use_linux)
    elif s.kind == ".i":
        _emit_print_cell(st, "@__.fmt_i", use_linux)
    elif s.kind == ".ir":
        _emit_print_cell(st, "@__.fmt_ir", use_linux)
    elif s.kind == ".f":
        _emit_print_cell(st, "@__.fmt_f", use_linux)
    elif s.kind == ".s":
        _emit_print_string(st, use_linux)
    elif s.kind == ",":
        _emit_getchar(st, use_linux)


def _emit_putchar(st: EmitState, use_linux: bool) -> None:
    p, ty = get_current_cell_ptr(st)
    v = "%" + st.ctx.next_temp("ov")
    st.lines.append(f"  {v} = load {ty}, ptr {p}, align {align(ty)}")
    if use_linux:
        b = "%" + st.ctx.next_temp("buf")
        st.alloca_lines.append(f"  {b} = alloca i8, align 1")
        if ty == "i8":
            st.lines.append(f"  store i8 {v}, ptr {b}, align 1")
        else:
            tv = "%" + st.ctx.next_temp("trunc")
            st.lines.append(f"  {tv} = trunc {ty} {v} to i8")
            st.lines.append(f"  store i8 {tv}, ptr {b}, align 1")
        st.lines.append(f"  call i64 @write(i32 1, ptr {b}, i64 1)")
    else:
        if ty == "i32":
            st.lines.append(f"  call i32 @putchar(i32 {v})")
        else:
            ev = "%" + st.ctx.next_temp("ext")
            st.lines.append(f"  {ev} = zext {ty} {v} to i32")
            st.lines.append(f"  call i32 @putchar(i32 {ev})")


def _emit_print_expr(st: EmitState, expr: A.Expr, fmt: str, use_linux: bool) -> None:
    if isinstance(expr, A.StringLit):
        s_id = st.get_string_ident(expr.value)
        if use_linux:
            ln = len(expr.value)
            st.lines.append(f"  call i64 @write(i32 1, ptr {s_id}, i64 {ln})")
        else:
            st.lines.append(f"  call i32 (ptr, ...) @printf(ptr {s_id})")
        return

    v, vty = emit_expr(st, expr, st.ctx)
    if use_linux:
        _emit_snprintf_write(st, v, vty, fmt)
    else:
        st.lines.append(f"  call i32 (ptr, ...) @printf(ptr {fmt}, {vty} {v})")


def _emit_print_cell(st: EmitState, fmt: str, use_linux: bool) -> None:
    p, ty = get_current_cell_ptr(st)
    v = "%" + st.ctx.next_temp("ov")
    st.lines.append(f"  {v} = load {ty}, ptr {p}, align {align(ty)}")
    if use_linux:
        _emit_snprintf_write(st, v, ty, fmt)
    else:
        st.lines.append(f"  call i32 (ptr, ...) @printf(ptr {fmt}, {ty} {v})")


def _emit_print_string(st: EmitState, use_linux: bool) -> None:
    p, ty = get_current_cell_ptr(st)
    v = "%" + st.ctx.next_temp("ov")
    st.lines.append(f"  {v} = load {ty}, ptr {p}, align {align(ty)}")
    if use_linux:
        ln = "%" + st.ctx.next_temp("len")
        st.lines.append(f"  {ln} = call i64 @strlen(ptr {v})")
        st.lines.append(f"  call i64 @write(i32 1, ptr {v}, i64 {ln})")
    else:
        st.lines.append(f"  call i32 (ptr, ...) @printf(ptr @__.fmt_s, {ty} {v})")


def _emit_getchar(st: EmitState, use_linux: bool) -> None:
    p, ty = get_current_cell_ptr(st)
    if use_linux:
        b = "%" + st.ctx.next_temp("buf")
        st.alloca_lines.append(f"  {b} = alloca i8, align 1")
        st.lines.append(f"  call i64 @read(i32 0, ptr {b}, i64 1)")
        rv = "%" + st.ctx.next_temp("rv")
        st.lines.append(f"  {rv} = load i8, ptr {b}, align 1")
        if ty == "i8":
            st.lines.append(f"  store i8 {rv}, ptr {p}, align 1")
        else:
            tv = "%" + st.ctx.next_temp("ext")
            st.lines.append(f"  {tv} = zext i8 {rv} to {ty}")
            st.lines.append(f"  store {ty} {tv}, ptr {p}, align {align(ty)}")
    else:
        r = "%" + st.ctx.next_temp("rv")
        st.lines.append(f"  {r} = call i32 @getchar()")
        if ty == "i32":
            st.lines.append(f"  store i32 {r}, ptr {p}, align 4")
        else:
            t = "%" + st.ctx.next_temp("tr")
            st.lines.append(f"  {t} = trunc i32 {r} to {ty}")
            st.lines.append(f"  store {ty} {t}, ptr {p}, align {align(ty)}")


def _emit_snprintf_write(st: EmitState, val: str, ty: str, fmt: str) -> None:
    buf = "%" + st.ctx.next_temp("sbuf")
    st.alloca_lines.append(f"  {buf} = alloca [64 x i8], align 1")
    p = "%" + st.ctx.next_temp("sp")
    st.lines.append(f"  {p} = getelementptr [64 x i8], ptr {buf}, i32 0, i32 0")
    st.lines.append(f"  call i32 (ptr, i64, ptr, ...) @snprintf(ptr {p}, i64 64, ptr {fmt}, {ty} {val})")
    ln = "%" + st.ctx.next_temp("slen")
    st.lines.append(f"  {ln} = call i64 @strlen(ptr {p})")
    st.lines.append(f"  call i64 @write(i32 1, ptr {p}, i64 {ln})")
