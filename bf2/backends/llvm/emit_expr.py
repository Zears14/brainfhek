"""Expression codegen: literals, identifiers, binary/unary ops, function calls."""

from __future__ import annotations

from typing import Tuple

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.emit_mem import gep, LLVMGenError
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.types import scalar_ty, align, LINUX_LIBC


def emit_expr(st: EmitState, e: A.Expr, ctx: LLVMContext) -> Tuple[str, str]:
    """Emit an expression and return (SSA value, LLVM type)."""
    if isinstance(e, A.IntLit):
        return (str(e.value), "i32")
    if isinstance(e, A.FloatLit):
        return (str(e.value), "double")
    if isinstance(e, A.BoolLit):
        return ("1" if e.value else "0", "i1")
    if isinstance(e, A.Ident):
        return _emit_ident(st, e, ctx)
    if isinstance(e, A.BinOp):
        return _emit_binop(st, e, ctx)
    if isinstance(e, A.Unary):
        return _emit_unary(st, e, ctx)
    if isinstance(e, A.Call):
        return emit_call_expr(st, e, ctx)
    if isinstance(e, A.RefExpr):
        return _emit_ref(st, e, ctx)
    raise LLVMGenError(f"unsupported expr {type(e)}")


def _emit_ident(st: EmitState, e: A.Ident, ctx: LLVMContext) -> Tuple[str, str]:
    if e.name in ctx.locals:
        p, ty = ctx.locals[e.name]
        t = "%" + ctx.next_temp(e.name)
        st.lines.append(f"  {t} = load {ty}, ptr {p}, align {align(ty)}")
        return (t, ty)
    if e.name in st.seg_slots:
        return (st.seg_slots[e.name], "ptr")
    raise LLVMGenError(f"Unknown identifier {e.name}", e.loc)


def _emit_binop(st: EmitState, e: A.BinOp, ctx: LLVMContext) -> Tuple[str, str]:
    l_v, l_t = emit_expr(st, e.left, ctx)
    r_v, r_t = emit_expr(st, e.right, ctx)

    # Determine result type
    # Priorities: double > float > i64 > i32 > i16 > i8
    is_d = l_t == "double" or r_t == "double"
    is_f = l_t == "float" or r_t == "float"

    def get_bw(t: str) -> int:
        return int(t[1:]) if t.startswith("i") else 0

    max_bw = max(get_bw(l_t), get_bw(r_t))

    if is_d:
        res_t = "double"
    elif is_f:
        res_t = "float"
    elif max_bw >= 64:
        res_t = "i64"
    elif max_bw >= 32:
        res_t = "i32"
    elif max_bw >= 16:
        res_t = "i16"
    else:
        res_t = "i8"

    # Coerce operands to result type
    l_v = coerce(st, l_v, l_t, res_t)
    r_v = coerce(st, r_v, r_t, res_t)

    is_fp = res_t in ("float", "double")

    if is_fp:
        op = {
            "+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv",
            "==": "fcmp oeq", "!=": "fcmp une",
            ">": "fcmp ogt", ">=": "fcmp oge",
            "<": "fcmp olt", "<=": "fcmp ole",
        }[e.op]
        t = "%" + ctx.next_temp("fbin")
        st.lines.append(f"  {t} = {op} {res_t} {l_v}, {r_v}")
        return (t, "i1" if e.op in ("==", "!=", ">", ">=", "<", "<=") else res_t)
    else:
        op = {
            "+": "add nsw", "-": "sub nsw", "*": "mul nsw", "/": "sdiv",
            "==": "icmp eq", "!=": "icmp ne",
            ">": "icmp sgt", ">=": "icmp sge",
            "<": "icmp slt", "<=": "icmp sle",
        }[e.op]
        t = "%" + ctx.next_temp("bin")
        st.lines.append(f"  {t} = {op} {res_t} {l_v}, {r_v}")
        return (t, "i1" if e.op in ("==", "!=", ">", ">=", "<", "<=") else res_t)


def _emit_unary(st: EmitState, e: A.Unary, ctx: LLVMContext) -> Tuple[str, str]:
    if e.op == "-":
        v, ty = emit_expr(st, e.expr, ctx)
        t = "%" + ctx.next_temp("neg")
        if ty in ("float", "double"):
            st.lines.append(f"  {t} = fneg {ty} {v}")
        else:
            st.lines.append(f"  {t} = sub nsw {ty} 0, {v}")
        return (t, ty)
    if e.op == "*":
        if not isinstance(e.expr, A.Ident):
            raise LLVMGenError("* only supports identifiers in SPEC", e.loc)
        p, _ = ctx.locals[e.expr.name]
        inn = ctx.ptr_inner.get(e.expr.name, A.TypeRef("i8"))
        pb = "%" + ctx.next_temp("pb")
        st.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
        ity = scalar_ty(inn)
        t = "%" + ctx.next_temp("dr")
        st.lines.append(f"  {t} = load {ity}, ptr {pb}, align {align(ity)}")
        return (t, ity)
    if e.op == "&":
        if not isinstance(e.expr, A.RefExpr):
            raise LLVMGenError("& expects a reference", e.loc)
        ptr, _ = gep(st, e.expr, ctx)
        return (ptr, "ptr")
    raise LLVMGenError(f"unsupported unary op: {e.op}", e.loc)


def _emit_ref(st: EmitState, e: A.RefExpr, ctx: LLVMContext) -> Tuple[str, str]:
    ptr, ty = gep(st, e, ctx)
    if e.parts and e.parts[0] == "@":
        return (ptr, "ptr")
    t = "%" + ctx.next_temp("ref")
    st.lines.append(f"  {t} = load {ty}, ptr {ptr}, align {align(ty)}")
    return (t, ty)


def emit_call_expr(st: EmitState, c: A.Call, ctx: LLVMContext) -> Tuple[str, str]:
    """Emit a function call expression and return (SSA value, LLVM type)."""
    if c.name == "sqrt":
        v, ty = emit_expr(st, c.args[0], ctx)
        t = "%" + ctx.next_temp("sqrt")
        st.lines.append(f"  {t} = call double @llvm.sqrt.f64(double {v})")
        return (t, "double")

    if getattr(st.mod, "use_linux_stdlib", False) and c.name in LINUX_LIBC:
        ret_t, arg_ts = LINUX_LIBC[c.name]
        parts = []
        for i, arg in enumerate(c.args):
            av, at = emit_expr(st, arg, ctx)
            if i < len(arg_ts):
                target_t = arg_ts[i]
                if target_t == "i64" and at == "i32":
                    cv = "%" + ctx.next_temp("zext")
                    st.lines.append(f"  {cv} = zext i32 {av} to i64")
                    av = cv
                    at = "i64"
            parts.append(f"{at} {av}")
        t = "%" + ctx.next_temp("call")
        if ret_t == "void":
            st.lines.append(f"  call void @{c.name}({', '.join(parts)})")
            return ("", "void")
        st.lines.append(f"  {t} = call {ret_t} @{c.name}({', '.join(parts)})")
        return (t, ret_t)

    fd = st.fns.get(c.name)
    if not fd:
        raise LLVMGenError(f"Unknown function {c.name}", c.loc)
    parts = []
    for i, (_, pt) in enumerate(fd.params):
        av, at = emit_expr(st, c.args[i], ctx)
        target_t = scalar_ty(pt)
        if target_t == "double" and at == "i32":
            cv = "%" + ctx.next_temp("conv")
            st.lines.append(f"  {cv} = sitofp i32 {av} to double")
            av = cv
            at = "double"
        parts.append(f"{at} {av}")
    rt = scalar_ty(fd.ret)
    if rt == "void":
        st.lines.append(f"  call void @{c.name}({', '.join(parts)})")
        return ("", "void")
    t = "%" + ctx.next_temp("call")
    st.lines.append(f"  {t} = call {rt} @{c.name}({', '.join(parts)})")
    return (t, rt)


def coerce(st: EmitState, val: str, from_ty: str, to_ty: str) -> str:
    """Emit a type coercion instruction if needed, returning the new SSA value."""
    if from_ty == to_ty:
        return val

    # Helper to get bitwidth of integer types
    def bitwidth(t: str) -> int:
        return int(t[1:]) if t.startswith("i") else 0

    fb, tb = bitwidth(from_ty), bitwidth(to_ty)

    # Int -> Int
    if fb and tb:
        if tb > fb:
            c = "%" + st.ctx.next_temp("zext")
            st.lines.append(f"  {c} = zext {from_ty} {val} to {to_ty}")
            return c
        else:
            c = "%" + st.ctx.next_temp("trunc")
            st.lines.append(f"  {c} = trunc {from_ty} {val} to {to_ty}")
            return c

    # Float/Double mappings
    is_f = from_ty in ("float", "double")
    is_t = to_ty in ("float", "double")

    # Float -> Float (fpext / fptrunc)
    if is_f and is_t:
        if from_ty == "float" and to_ty == "double":
            c = "%" + st.ctx.next_temp("fpext")
            st.lines.append(f"  {c} = fpext float {val} to double")
            return c
        else:
            c = "%" + st.ctx.next_temp("fptrunc")
            st.lines.append(f"  {c} = fptrunc double {val} to float")
            return c

    # Int -> Float/Double
    if fb and is_t:
        c = "%" + st.ctx.next_temp("sitofp")
        st.lines.append(f"  {c} = sitofp {from_ty} {val} to {to_ty}")
        return c

    # Float/Double -> Int
    if is_f and tb:
        c = "%" + st.ctx.next_temp("fptosi")
        st.lines.append(f"  {c} = fptosi {from_ty} {val} to {to_ty}")
        return c

    return val
