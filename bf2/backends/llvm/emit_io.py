"""IO statement codegen: `.`, `.i`, `.f`, `.s`, `,` handlers."""

from __future__ import annotations

from llvmlite import ir

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.emit_mem import get_current_cell_ptr
from bf2.backends.llvm.emit_expr import emit_expr
from bf2.backends.llvm.types import align, Int8, Int32, Int64, Pointer


def emit_io(st: EmitState, s: A.IOStmt) -> None:
    """Emit IO operations for all `.` and `,` variants."""
    use_linux = getattr(st.mod, "use_linux_stdlib", False)

    if s.kind == ".":
        _emit_putchar(st, use_linux)
    elif s.kind == ".i" and s.expr:
        _emit_print_expr(st, s.expr, "__bf2_fmt_i", use_linux)
    elif s.kind == ".ir" and s.expr:
        _emit_print_expr(st, s.expr, "__bf2_fmt_ir", use_linux)
    elif s.kind == ".i64" and s.expr:
        _emit_print_expr(st, s.expr, "__bf2_fmt_i64", use_linux)
    elif s.kind == ".i64r" and s.expr:
        _emit_print_expr(st, s.expr, "__bf2_fmt_i64r", use_linux)
    elif (s.kind == ".f" or s.kind == ".fr") and s.expr:
        fmt = "__bf2_fmt_f" if s.kind == ".f" else "__bf2_fmt_fr"
        _emit_print_expr(st, s.expr, fmt, use_linux)
    elif s.kind == ".s" and s.expr:
        _emit_print_expr(st, s.expr, "__bf2_fmt_s", use_linux)
    elif s.kind == ".i":
        _emit_print_cell(st, "__bf2_fmt_i", use_linux)
    elif s.kind == ".ir":
        _emit_print_cell(st, "__bf2_fmt_ir", use_linux)
    elif s.kind == ".f":
        _emit_print_cell(st, "__bf2_fmt_f", use_linux)
    elif s.kind == ".s":
        _emit_print_string(st, use_linux)
    elif s.kind == ",":
        _emit_getchar(st, use_linux)


def _emit_putchar(st: EmitState, use_linux: bool) -> None:
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    v = ctx.builder.load(p, name=ctx.next_temp("ov"))

    if use_linux:
        buf = ctx.hoist_alloca(Int8, name=ctx.next_temp("buf"))
        if isinstance(ty, ir.IntType) and ty.width > 8:
            tv = ctx.builder.trunc(v, Int8, name=ctx.next_temp("trunc"))
            ctx.builder.store(tv, buf)
        else:
            ctx.builder.store(v, buf)
        write_fn = _get_or_declare(st, "write", ir.FunctionType(Int64, [Int32, Pointer, Int64]))
        ctx.builder.call(write_fn, [
            ir.Constant(Int32, 1),
            buf,
            ir.Constant(Int64, 1)
        ])
    else:
        putchar_fn = _get_or_declare(st, "putchar", ir.FunctionType(Int32, [Int32]))
        if isinstance(ty, ir.IntType):
            if ty.width == 32:
                ctx.builder.call(putchar_fn, [v])
            elif ty.width < 32:
                ev = ctx.builder.zext(v, Int32, name=ctx.next_temp("ext"))
                ctx.builder.call(putchar_fn, [ev])
            else:
                ev = ctx.builder.trunc(v, Int32, name=ctx.next_temp("trunc"))
                ctx.builder.call(putchar_fn, [ev])
        else:
            ev = ctx.builder.fptosi(v, Int32, name=ctx.next_temp("ext"))
            ctx.builder.call(putchar_fn, [ev])


def _get_or_declare(st: EmitState, name: str, fnty: ir.FunctionType) -> ir.Function:
    fn = st.module.globals.get(name)
    if fn is None:
        fn = ir.Function(st.module, fnty, name=name)
    return fn


def _get_or_create_string_constant(st: EmitState, value: str) -> ir.GlobalVariable:
    name = st.get_string_ident(value)
    gv = st.module.globals.get(name)
    if gv is not None:
        return gv

    const_arr = ir.ArrayType(Int8, len(value) + 1)
    byte_arr = bytearray(value.encode("utf-8")) + b"\0"
    gv = ir.GlobalVariable(st.module, const_arr, name=name)
    gv.initializer = ir.Constant(const_arr, byte_arr)
    gv.align = 1
    gv.linkage = "private"
    gv.unnamed_addr = True
    return gv


def _emit_print_expr(st: EmitState, expr: A.Expr, fmt: str, use_linux: bool) -> None:
    ctx = st.ctx
    if isinstance(expr, A.StringLit):
        s_gv = _get_or_create_string_constant(st, expr.value)
        if use_linux:
            ln = len(expr.value)
            write_fn = _get_or_declare(st, "write", ir.FunctionType(Int64, [Int32, Pointer, Int64]))
            s_cast = ctx.builder.bitcast(s_gv, Pointer, name=ctx.next_temp("scast"))
            ctx.builder.call(write_fn, [
                ir.Constant(Int32, 1),
                s_cast,
                ir.Constant(Int64, ln)
            ])
        else:
            s_cast = ctx.builder.bitcast(s_gv, Pointer, name=ctx.next_temp("scast"))
            printf_fn = _get_or_declare(st, "printf", ir.FunctionType(Int32, [Pointer], var_arg=True))
            ctx.builder.call(printf_fn, [s_cast])
        return

    v, vty = emit_expr(st, expr, ctx)
    if use_linux:
        _emit_snprintf_write(st, v, vty, fmt)
    else:
        fmt_gv = st.module.globals.get(fmt)
        if fmt_gv is None:
            return
        fmt_cast = ctx.builder.bitcast(fmt_gv, Pointer, name=ctx.next_temp("fmtcast"))
        printf_fn = _get_or_declare(st, "printf", ir.FunctionType(Int32, [Pointer], var_arg=True))
        ctx.builder.call(printf_fn, [fmt_cast, v])


def _emit_print_cell(st: EmitState, fmt: str, use_linux: bool) -> None:
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    v = ctx.builder.load(p, align=align(ty), name=ctx.next_temp("ov"))
    if use_linux:
        _emit_snprintf_write(st, v, ty, fmt)
    else:
        fmt_gv = st.module.globals.get(fmt)
        printf_fn = _get_or_declare(st, "printf", ir.FunctionType(Int32, [Pointer], var_arg=True))
        ctx.builder.call(printf_fn, [fmt_gv, v])


def _emit_print_string(st: EmitState, use_linux: bool) -> None:
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    v = ctx.builder.load(p, align=align(ty), name=ctx.next_temp("ov"))
    if use_linux:
        strlen_fn = _get_or_declare(st, "strlen", ir.FunctionType(Int64, [Pointer]))
        ln = ctx.builder.call(strlen_fn, [v], name=ctx.next_temp("len"))
        write_fn = _get_or_declare(st, "write", ir.FunctionType(Int64, [Int32, Pointer, Int64]))
        ctx.builder.call(write_fn, [ir.Constant(Int32, 1), v, ln])
    else:
        fmt_gv = st.module.globals.get("__bf2_fmt_s")
        printf_fn = _get_or_declare(st, "printf", ir.FunctionType(Int32, [Pointer], var_arg=True))
        ctx.builder.call(printf_fn, [fmt_gv, v])


def _emit_getchar(st: EmitState, use_linux: bool) -> None:
    ctx = st.ctx
    p, ty = get_current_cell_ptr(st)
    if use_linux:
        buf = ctx.hoist_alloca(Int8, name=ctx.next_temp("buf"))
        read_fn = _get_or_declare(st, "read", ir.FunctionType(Int64, [Int32, Pointer, Int64]))
        ctx.builder.call(read_fn, [ir.Constant(Int32, 0), buf, ir.Constant(Int64, 1)])
        rv = ctx.builder.load(buf, align=1, name=ctx.next_temp("rv"))
        if ty == Int8:
            ctx.builder.store(rv, p, align=1)
        else:
            tv = ctx.builder.zext(rv, ty, name=ctx.next_temp("ext"))
            ctx.builder.store(tv, p, align=align(ty))
    else:
        getchar_fn = _get_or_declare(st, "getchar", ir.FunctionType(Int32, []))
        r = ctx.builder.call(getchar_fn, [], name=ctx.next_temp("rv"))
        if ty == Int32:
            ctx.builder.store(r, p, align=4)
        else:
            t = ctx.builder.trunc(r, ty, name=ctx.next_temp("tr"))
            ctx.builder.store(t, p, align=align(ty))


def _emit_snprintf_write(st: EmitState, val: ir.Value, ty: ir.Type, fmt: str) -> None:
    ctx = st.ctx
    buf = ctx.hoist_alloca(ir.ArrayType(Int8, 64), name=ctx.next_temp("sbuf"))
    p = ctx.builder.gep(buf, [ir.Constant(Int32, 0), ir.Constant(Int32, 0)], name=ctx.next_temp("sp"))
    fmt_gv = st.module.globals.get(fmt)
    if fmt_gv is None:
        return
    snprintf_fn = _get_or_declare(st, "snprintf", ir.FunctionType(Int32, [Pointer, Int64, Pointer], var_arg=True))
    p_cast = ctx.builder.bitcast(p, Pointer, name=ctx.next_temp("pcast"))
    fmt_cast = ctx.builder.bitcast(fmt_gv, Pointer, name=ctx.next_temp("fmtcast"))
    args = [p_cast, ir.Constant(Int64, 64), fmt_cast]
    if ty == Int32:
        args.append(ctx.builder.sext(val, Int64))
    elif ty == Int8:
        args.append(ctx.builder.zext(val, Int64))
    else:
        args.append(val)
    ctx.builder.call(snprintf_fn, args)
    strlen_fn = _get_or_declare(st, "strlen", ir.FunctionType(Int64, [Pointer]))
    ln = ctx.builder.call(strlen_fn, [p], name=ctx.next_temp("slen"))
    write_fn = _get_or_declare(st, "write", ir.FunctionType(Int64, [Int32, Pointer, Int64]))
    ctx.builder.call(write_fn, [ir.Constant(Int32, 1), p, ln])
