from __future__ import annotations

from typing import Any, Dict

from bf2 import ast_nodes as A
from bf2.errors import BF2TypeError, SourceLoc
from bf2.memory import build_struct_layout


class TypeEnv:
    def __init__(self) -> None:
        self.structs: Dict[str, A.StructDecl] = {}
        self.segments: Dict[str, A.TypeRef] = {}
        self.funcs: Dict[str, tuple[list[tuple[str, A.TypeRef]], A.TypeRef]] = {}
        self.locals: Dict[str, A.TypeRef] = {}
        self.use_linux_stdlib: bool = False


def _infer_linux_libc_call(c: A.Call, env: TypeEnv) -> A.TypeRef:
    """Type-check calls declared by #include \"stdlib\" (Linux libc)."""
    n = c.name
    args = c.args
    if n in ("write", "read"):
        if len(args) != 3:
            raise BF2TypeError(f"{n} expects 3 arguments", _loc(c))
        for a in args:
            _infer_expr(a, env)
        return A.TypeRef("i64")
    if n == "open":
        if len(args) not in (2, 3):
            raise BF2TypeError("open expects 2 or 3 arguments", _loc(c))
        for a in args:
            _infer_expr(a, env)
        return A.TypeRef("i32")
    if n == "close":
        if len(args) != 1:
            raise BF2TypeError("close expects 1 argument", _loc(c))
        _infer_expr(args[0], env)
        return A.TypeRef("i32")
    if n == "exit":
        if len(args) != 1:
            raise BF2TypeError("exit expects 1 argument", _loc(c))
        _infer_expr(args[0], env)
        return A.TypeRef("void")
    if n == "fork":
        if len(args) != 0:
            raise BF2TypeError("fork expects no arguments", _loc(c))
        return A.TypeRef("i32")
    if n == "getpid":
        if len(args) != 0:
            raise BF2TypeError("getpid expects no arguments", _loc(c))
        return A.TypeRef("i32")
    if n == "strlen":
        if len(args) != 1:
            raise BF2TypeError("strlen expects 1 argument", _loc(c))
        _infer_expr(args[0], env)
        return A.TypeRef("i64")
    if n == "nanosleep":
        if len(args) != 2:
            raise BF2TypeError("nanosleep expects 2 arguments", _loc(c))
        for a in args:
            _infer_expr(a, env)
        return A.TypeRef("i32")
    raise BF2TypeError(f"unknown libc function {n!r}", _loc(c))


def _loc(x: Any) -> SourceLoc:
    return getattr(x, "loc", SourceLoc(1, 1))


def check_module(mod: A.Module) -> None:
    env = TypeEnv()
    env.use_linux_stdlib = getattr(mod, "use_linux_stdlib", False)
    for it in mod.items:
        if isinstance(it, A.StructDecl):
            env.structs[it.name] = it
        elif isinstance(it, A.SegmentDecl):
            env.segments[it.name] = it.elem_type
        elif isinstance(it, A.FunctionDef):
            env.funcs[it.name] = (list(it.params), it.ret)
    for it in mod.items:
        if isinstance(it, A.FunctionDef):
            le = TypeEnv()
            le.structs = env.structs
            le.segments = env.segments
            le.funcs = env.funcs
            le.use_linux_stdlib = env.use_linux_stdlib
            for n, t in it.params:
                le.locals[n] = t
            _check_block(it.body, le, it.ret)
        elif isinstance(it, A.ReactorDef):
            le = TypeEnv()
            le.structs = env.structs
            le.segments = env.segments
            le.funcs = env.funcs
            le.use_linux_stdlib = env.use_linux_stdlib
            _check_block(it.body, le, A.TypeRef("i32"))


def _check_block(b: A.Block, env: TypeEnv, ret: A.TypeRef) -> None:
    for st in b.stmts:
        _check_stmt(st, env, ret)


def _check_stmt(st: Any, env: TypeEnv, ret: A.TypeRef) -> None:
    if isinstance(st, (A.SegmentStmt, A.StructStmt)):
        return
    if isinstance(st, A.VarDecl):
        t = _infer_expr(st.init, env) if st.init else st.ty
        if st.ty.name != t.name and not (st.ty.name == "f64" and t.name in ("i32", "f32")):
            pass
        env.locals[st.name] = st.ty
        return
    if isinstance(st, A.PtrDecl):
        env.locals[st.name] = A.TypeRef("ptr", st.inner)
        return
    if isinstance(st, A.AssignStmt):
        _infer_expr(st.rhs, env)
        return
    if isinstance(st, A.IfStmt):
        _check_block(st.then, env, ret)
        if st.els:
            _check_block(st.els, env, ret)
        return
    if isinstance(st, (A.LoopBF, A.LoopCounted)):
        _check_block(st.body, env, ret)
        return
    if isinstance(st, A.RetStmt):
        if st.value is None:
            return
        _infer_expr(st.value, env)
        return
    if isinstance(st, A.CallStmt):
        _infer_call(st.call, env)
        return
    if isinstance(st, A.ExprStmt):
        _infer_expr(st.expr, env)
        return
    if isinstance(
        st,
        (
            A.LoadOp,
            A.StoreOp,
            A.SwapOp,
            A.MoveOp,
            A.MoveRel,
            A.CellArith,
            A.CellArithRef,
            A.CellAssignLit,
            A.IOStmt,
            A.LabelStmt,
            A.JumpStmt,
            A.PtrArith,
            A.PtrRead,
        ),
    ):
        if isinstance(st, A.IOStmt) and st.expr is not None:
            _infer_expr(st.expr, env)
        return
    if isinstance(st, A.PtrWrite):
        _infer_expr(st.value, env)
        return
    if isinstance(st, A.AllocStmt):
        if st.name:
            env.locals[st.name] = A.TypeRef("ptr", st.ty)
        return
    if isinstance(st, A.FreeStmt):
        return
    raise BF2TypeError(f"unsupported statement: {type(st).__name__}", _loc(st))


def _infer_call(c: A.Call, env: TypeEnv) -> A.TypeRef:
    if c.name not in env.funcs:
        if env.use_linux_stdlib and c.name in (
            "write",
            "read",
            "open",
            "close",
            "exit",
            "fork",
            "getpid",
            "strlen",
            "nanosleep",
        ):
            return _infer_linux_libc_call(c, env)
        if c.name in ("sqrt",):
            for a in c.args:
                _infer_expr(a, env)
            return A.TypeRef("f64")
        raise BF2TypeError(f"unknown function {c.name}", _loc(c))
    params, rt = env.funcs[c.name]
    if len(params) != len(c.args):
        raise BF2TypeError("arg count mismatch", _loc(c))
    for (_, pt), ae in zip(params, c.args):
        at = _infer_expr(ae, env)
        if pt.name != at.name and not (pt.name == "ptr" and at.name == "ptr"):
            if pt.name in env.structs and at.name == pt.name:
                continue
            pass
    return rt


def _infer_expr(e: A.Expr, env: TypeEnv) -> A.TypeRef:
    if isinstance(e, A.IntLit):
        return A.TypeRef("i32")
    if isinstance(e, A.FloatLit):
        return A.TypeRef("f64")
    if isinstance(e, A.BoolLit):
        return A.TypeRef("bool")
    if isinstance(e, A.Ident):
        if e.name not in env.locals:
            return A.TypeRef("i32")
        return env.locals[e.name]
    if isinstance(e, A.RefExpr):
        return _type_ref_expr(e, env)
    if isinstance(e, A.BinOp):
        _infer_expr(e.left, env)
        _infer_expr(e.right, env)
        return A.TypeRef("i32")
    if isinstance(e, A.Unary):
        if e.op == "&":
            return A.TypeRef("ptr", _infer_expr(e.expr, env))  # type: ignore
        if e.op == "*":
            t = _infer_expr(e.expr, env)
            if t.name == "ptr" and t.inner:
                return t.inner
            return A.TypeRef("i32")
        _infer_expr(e.expr, env)
        return A.TypeRef("i32")
    if isinstance(e, A.Call):
        return _infer_call(e, env)
    raise BF2TypeError("bad expr", _loc(e))


def _type_ref_expr(r: A.RefExpr, env: TypeEnv) -> A.TypeRef:
    parts = r.parts
    if parts and parts[0] == "@":
        return A.TypeRef("i8")
    if not parts:
        raise BF2TypeError("bad ref", _loc(r))
    head = str(parts[0])
    if head in env.locals:
        t = env.locals[head]
        slot = 1
        while slot < len(parts):
            p = parts[slot]
            if isinstance(p, str):
                if t.name == "ptr" and t.inner:
                    inn = t.inner
                    if inn.name in env.structs:
                        sl = build_struct_layout(env.structs[inn.name])
                        for fn, ft in sl.fields:
                            if fn == p:
                                t = ft
                                break
                elif t.name in env.structs:
                    sl = build_struct_layout(env.structs[t.name])
                    for fn, ft in sl.fields:
                        if fn == p:
                            t = ft
                            break
            slot += 1
        return t
    if head in env.segments:
        return env.segments[head]
    return A.TypeRef("i32")
