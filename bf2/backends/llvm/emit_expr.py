"""Expression codegen: literals, identifiers, binary/unary ops, function calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from llvmlite import ir

from bf2.core import ast as A
from bf2.backends.llvm.emit_state import EmitState
from bf2.backends.llvm.emit_mem import gep, LLVMGenError
from bf2.backends.llvm.context import LLVMContext
from bf2.backends.llvm.types import (
    to_ir_type, align, LINUX_LIBC,
    Int1, Int8, Int32, Int64, Float, Double, Pointer
)

if TYPE_CHECKING:
    from bf2.backends.llvm.types import ir


def emit_expr(st: EmitState, e: A.Expr, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    """Emit an expression and return (ir.Value, ir.Type)."""
    if isinstance(e, A.IntLit):
        return (ir.Constant(Int32, e.value), Int32)
    if isinstance(e, A.FloatLit):
        return (ir.Constant(Double, e.value), Double)
    if isinstance(e, A.BoolLit):
        return (ir.Constant(Int1, 1 if e.value else 0), Int1)
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
    if isinstance(e, A.StringLit):
        return _emit_string_lit(st, e, ctx)
    raise LLVMGenError(f"unsupported expr {type(e)}")


def _emit_ident(st: EmitState, e: A.Ident, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    if e.name in ctx.locals:
        ptr, ty = ctx.locals[e.name]
        loaded = ctx.builder.load(ptr, align=align(ty))
        return (loaded, ty)
    if e.name in st.seg_slots:
        gv = st.seg_slots[e.name]
        return (gv, gv.type.pointee)
    raise LLVMGenError(f"Unknown identifier {e.name}", e.loc)


def _emit_binop(st: EmitState, e: A.BinOp, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    l_v, l_t = emit_expr(st, e.left, ctx)
    r_v, r_t = emit_expr(st, e.right, ctx)

    is_d = l_t == Double or r_t == Double
    is_f = l_t == Float or r_t == Float or l_t == Double or r_t == Double

    def get_bw(t: ir.Type) -> int:
        if isinstance(t, ir.IntType):
            return t.width
        return 0

    max_bw = max(get_bw(l_t), get_bw(r_t))

    if is_d:
        res_t = Double
    elif is_f:
        res_t = Float
    elif max_bw >= 64:
        res_t = Int64
    elif max_bw >= 32:
        res_t = Int32
    elif max_bw >= 16:
        res_t = ir.IntType(16)
    else:
        res_t = Int8

    l_v = _coerce(st, l_v, l_t, res_t, ctx)
    r_v = _coerce(st, r_v, r_t, res_t, ctx)

    is_fp = res_t in (Float, Double)

    if is_fp:
        fp_ops = {
            "+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv",
        }
        cmp_ops = {
            "==": "fcmp oeq", "!=": "fcmp une",
            ">": "fcmp ogt", ">=": "fcmp oge",
            "<": "fcmp olt", "<=": "fcmp ole",
        }
        if e.op in fp_ops:
            method = fp_ops[e.op]
            result = getattr(ctx.builder, method)(l_v, r_v, name=ctx.next_temp("fbin"))
            return (result, res_t)
        else:
            op = cmp_ops[e.op]
            result = ctx.builder.icmp(op, l_v, r_v, name=ctx.next_temp("fcmp"))
            return (result, Int1)
    else:
        int_ops = {
            "+": "add", "-": "sub", "*": "mul", "/": "sdiv",
        }
        cmp_ops = {
            "==": "==", "!=": "!=",
            ">": ">", ">=": ">=",
            "<": "<", "<=": "<=",
        }
        if e.op in int_ops:
            method = int_ops[e.op]
            result = getattr(ctx.builder, method)(l_v, r_v, name=ctx.next_temp("bin"), flags=["nsw"])
            return (result, res_t)
        else:
            op = cmp_ops[e.op]
            result = ctx.builder.icmp_signed(op, l_v, r_v, name=ctx.next_temp("icmp"))
            return (result, Int1)


def _emit_unary(st: EmitState, e: A.Unary, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    if e.op == "-":
        v, ty = emit_expr(st, e.expr, ctx)
        if ty in (Float, Double):
            result = ctx.builder.fneg(v, name=ctx.next_temp("neg"))
        else:
            zero = ir.Constant(ty, 0)
            result = ctx.builder.sub(zero, v, name=ctx.next_temp("neg"), flags=["nsw"])
        return (result, ty)
    if e.op == "*":
        if not isinstance(e.expr, A.Ident):
            raise LLVMGenError("* only supports identifiers in SPEC", e.loc)
        ptr, _ = ctx.locals[e.expr.name]
        inn = ctx.ptr_inner.get(e.expr.name, A.TypeRef("i8"))
        pb = ctx.builder.load(ptr, align=8, name=ctx.next_temp("pb"))
        ity = to_ir_type(inn)
        if pb.type != ir.PointerType(ity):
            pb = ctx.builder.bitcast(pb, ir.PointerType(ity), name=ctx.next_temp("pbc"))
        result = ctx.builder.load(pb, align=align(ity), name=ctx.next_temp("dr"))
        return (result, ity)
    if e.op == "&":
        if not isinstance(e.expr, A.RefExpr):
            raise LLVMGenError("& expects a reference", e.loc)
        ptr, _ = gep(st, e.expr, ctx)
        return (ptr, ptr.type)
    raise LLVMGenError(f"unsupported unary op: {e.op}", e.loc)


def _emit_ref(st: EmitState, e: A.RefExpr, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    ptr, ty = gep(st, e, ctx)
    if e.parts and e.parts[0] == "@":
        return (ptr, ptr.type)
    loaded = ctx.builder.load(ptr, align=align(ty), name=ctx.next_temp("ref"))
    return (loaded, ty)


def _emit_string_lit(st: EmitState, e: A.StringLit, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    name = st.get_string_ident(e.value)
    gv = st.module.globals.get(name)
    if gv is None:
        const_arr = ir.ArrayType(Int8, len(e.value) + 1)
        gv = ir.GlobalVariable(st.module, const_arr, name=name)
    ptr = ctx.builder.bitcast(gv, Pointer, name=ctx.next_temp("str"))
    return (ptr, Pointer)


def emit_call_expr(st: EmitState, c: A.Call, ctx: LLVMContext) -> Tuple[ir.Value, ir.Type]:
    """Emit a function call expression and return (ir.Value, ir.Type)."""
    if c.name == "sqrt":
        v, ty = emit_expr(st, c.args[0], ctx)
        sqrt_fn = st.module.globals.get("llvm.sqrt.f64")
        if sqrt_fn is None:
            sqrt_ty = ir.FunctionType(Double, [Double])
            sqrt_fn = ir.Function(st.module, sqrt_ty, name="llvm.sqrt.f64")
        result = ctx.builder.call(sqrt_fn, [v], name=ctx.next_temp("sqrt"))
        return (result, Double)

    if c.name in LINUX_LIBC:
        ret_name, arg_names = LINUX_LIBC[c.name]
        rt = get_linux_type(ret_name)
        arg_tys = [get_linux_type(a) for a in arg_names if a != "..."]
        fn = st.module.globals.get(c.name)
        if fn is None:
            fnty = ir.FunctionType(rt, arg_tys, var_arg="..." in arg_names)
            fn = ir.Function(st.module, fnty, name=c.name)

        args = []
        for i, target_t in enumerate(arg_tys):
            av, at = emit_expr(st, c.args[i], ctx)
            av = _coerce(st, av, at, target_t, ctx)
            if isinstance(av.type, ir.PointerType) and av.type != target_t and isinstance(target_t, ir.PointerType):
                av = ctx.builder.bitcast(av, target_t, name=ctx.next_temp("bc"))
            args.append(av)

        if rt == ir.VoidType():
            ctx.builder.call(fn, args)
            return (None, ir.VoidType())

        result = ctx.builder.call(fn, args, name=ctx.next_temp("call"))
        return (result, rt)

    fd = st.fns.get(c.name)
    if not fd:
        raise LLVMGenError(f"Unknown function {c.name}", c.loc)

    args = []
    for i, (_, pt) in enumerate(fd.params):
        av, at = emit_expr(st, c.args[i], ctx)
        target_t = to_ir_type(pt)
        # Handle integer type coercion (i32 <-> i64)
        if target_t == Int64 and at == Int32:
            av = ctx.builder.sext(av, Int64, name=ctx.next_temp("sext"))
            at = Int64
        elif at == Int64 and target_t == Int32:
            av = ctx.builder.trunc(av, Int32, name=ctx.next_temp("trunc"))
            at = Int32
        elif target_t == Double and at == Int32:
            av = ctx.builder.sitofp(av, Double, name=ctx.next_temp("conv"))
            at = Double
        elif at == Double and target_t == Int32:
            av = ctx.builder.fptosi(av, Int32, name=ctx.next_temp("conv"))
            at = Int32
        elif isinstance(at, ir.PointerType) and isinstance(target_t, ir.PointerType) and at != target_t:
            av = ctx.builder.bitcast(av, target_t, name=ctx.next_temp("bc"))
        args.append(av)
    rt = to_ir_type(fd.ret)
    fn = st.module.globals.get(fd.name)
    if fn is None:
        arg_tys = [to_ir_type(pt) for _, pt in fd.params]
        fnty = ir.FunctionType(rt, arg_tys)
        fn = ir.Function(st.module, fnty, name=fd.name)
    if rt == ir.VoidType():
        ctx.builder.call(fn, args)
        return (None, ir.VoidType())
    result = ctx.builder.call(fn, args, name=ctx.next_temp("call"))
    return (result, rt)


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
    if name == "...":
        return ir.VarArgType()
    return Int32


def _coerce(st: EmitState, val: ir.Value, from_ty: ir.Type, to_ty: ir.Type, ctx: LLVMContext) -> ir.Value:
    """Emit a type coercion instruction if needed, returning the new ir.Value."""
    if from_ty == to_ty:
        return val

    if isinstance(from_ty, ir.IntType) and isinstance(to_ty, ir.IntType):
        if to_ty.width > from_ty.width:
            return ctx.builder.sext(val, to_ty, name=ctx.next_temp("sext"))
        else:
            return ctx.builder.trunc(val, to_ty, name=ctx.next_temp("trunc"))

    is_f = isinstance(from_ty, (ir.FloatType, ir.DoubleType))
    is_t = isinstance(to_ty, (ir.FloatType, ir.DoubleType))

    if is_f and is_t:
        if isinstance(from_ty, ir.FloatType) and isinstance(to_ty, ir.DoubleType):
            return ctx.builder.fpext(val, to_ty, name=ctx.next_temp("fpext"))
        else:
            return ctx.builder.fptrunc(val, to_ty, name=ctx.next_temp("fptrunc"))

    if isinstance(from_ty, ir.IntType) and is_t:
        return ctx.builder.sitofp(val, to_ty, name=ctx.next_temp("sitofp"))

    if is_f and isinstance(to_ty, ir.IntType):
        return ctx.builder.fptosi(val, to_ty, name=ctx.next_temp("fptosi"))

    if isinstance(from_ty, ir.PointerType) and isinstance(to_ty, ir.PointerType):
        return ctx.builder.bitcast(val, to_ty, name=ctx.next_temp("bitcast"))

    return val


def coerce(st: EmitState, val: str, from_ty: str, to_ty: str) -> str:
    """Legacy compatibility function - returns the value as-is (for string-based code)."""
    return val
