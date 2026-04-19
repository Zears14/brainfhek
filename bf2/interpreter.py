from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from bf2 import ast_nodes as A
from bf2.errors import BF2RuntimeError, SourceLoc
from bf2.memory import Pointer, SegmentTable, StructLayout, build_struct_layout, watch_key
from bf2.reactor import ReactorEngine


@dataclass
class Frame:
    locals: Dict[str, Any] = field(default_factory=dict)
    cursor_seg: str = "__bf"
    cursor_slot: int = 0
    returned: bool = False


class Interpreter:
    def __init__(self, mod: A.Module):
        self.mod = mod
        self.structs: Dict[str, StructLayout] = {}
        self.table: SegmentTable = SegmentTable({})
        self.reactors = ReactorEngine()
        self.out: List[str] = []
        self.labels: Dict[str, int] = {}
        self._stdin_buf: str = ""
        self._heap_seq = 0

    def _ensure_bf(self) -> None:
        if "__bf" not in self.table.segments:
            from bf2.ast_nodes import SegmentDecl, TypeRef

            d = SegmentDecl("__bf", TypeRef("i8"), 30000, SourceLoc(1, 1))
            self.table.add_segment(d, None)

    def run(self) -> None:
        self._ensure_bf()
        for it in self.mod.items:
            if isinstance(it, A.StructDecl):
                self.structs[it.name] = build_struct_layout(it)
            elif isinstance(it, A.SegmentDecl):
                lay = self.structs.get(it.elem_type.name) if it.elem_type.name in self.structs else None
                self.table.add_segment(it, lay)
            elif isinstance(it, A.SegmentStmt):
                lay = self.structs.get(it.decl.elem_type.name) if it.decl.elem_type.name in self.structs else None
                self.table.add_segment(it.decl, lay)
            elif isinstance(it, A.StructStmt):
                self.structs[it.decl.name] = build_struct_layout(it.decl)
        for it in self.mod.items:
            if isinstance(it, A.ReactorDef):
                self._reg_watch(it)
        for it in self.mod.items:
            if isinstance(it, A.FunctionDef) and it.name == "main":
                self.call_fn(it, [])
                return
        raise BF2RuntimeError("no main()", SourceLoc(1, 1))

    def _reg_watch(self, w: A.ReactorDef) -> None:
        seg, slot = self._resolve_ref_static(w.target.parts)
        key = watch_key(seg, slot)

        def _cb(_new_val: Any) -> None:
            fr = Frame(cursor_seg=seg, cursor_slot=slot)
            self._exec_block(w.body, fr)

        self.reactors.register(key, _cb)

    def _resolve_ref_static(self, parts: List[Any]) -> Tuple[str, int]:
        return self.table.resolve_ref(parts)

    def call_fn(self, fn: A.FunctionDef, args: List[Any]) -> Any:
        fr = Frame(returned=False)
        for (pn, _pt), av in zip(fn.params, args):
            fr.locals[pn] = av
        if fn.params:
            p0 = fn.params[0][1]
            if p0.name in ("i8", "i16", "i32", "i64"):
                self._ensure_cp(fr)
                self.table.write_slot("__cp", 0, int(args[0]))
                fr.cursor_seg, fr.cursor_slot = "__cp", 0
        self._exec_block(fn.body, fr)
        return fr.locals.get("__ret", 0)

    def _ensure_cp(self, fr: Frame) -> None:
        if "__cp" not in self.table.segments:
            from bf2.ast_nodes import SegmentDecl, TypeRef

            self.table.add_segment(SegmentDecl("__cp", TypeRef("i32"), 1, SourceLoc(1, 1)), None)

    def _exec_block(self, block: A.Block, fr: Frame) -> None:
        for st in block.stmts:
            self._exec_stmt(st, fr)
            if fr.returned:
                break

    def _exec_stmt(self, st: Any, fr: Frame) -> None:
        if isinstance(st, A.SegmentStmt):
            lay = self.structs.get(st.decl.elem_type.name) if st.decl.elem_type.name in self.structs else None
            self.table.add_segment(st.decl, lay)
            return
        if isinstance(st, A.StructStmt):
            self.structs[st.decl.name] = build_struct_layout(st.decl)
            self.table.structs[st.decl.name] = self.structs[st.decl.name]
            return
        if isinstance(st, A.MoveOp):
            seg, slot = self._resolve_ref(fr, st.target.parts)
            fr.cursor_seg, fr.cursor_slot = seg, slot
            return
        if isinstance(st, A.MoveRel):
            fr.cursor_slot += st.delta
            return
        if isinstance(st, A.CellArith):
            v = self._read_cell(fr)
            a = st.amount if st.amount is not None else 1
            if st.op == "+":
                nv = v + a
            elif st.op == "-":
                nv = v - a
            elif st.op == "*":
                nv = v * a
            elif st.op == "/":
                nv = int(v // a) if a else 0
            else:
                nv = v
            self._write_cell(fr, nv)
            return
        if isinstance(st, A.CellArithRef):
            seg, slot = self._resolve_ref(fr, st.target.parts)
            v = self.table.read_slot(seg, slot)
            a = st.amount
            if st.op == "+":
                nv = v + a
            elif st.op == "-":
                nv = v - a
            elif st.op == "*":
                nv = v * a
            elif st.op == "/":
                nv = int(v // a) if a else 0
            else:
                nv = v
            self._write_slot(seg, slot, nv)
            return
        if isinstance(st, A.CellAssignLit):
            self._write_cell(fr, st.value)
            return
        if isinstance(st, A.LoadOp):
            seg, slot = self._resolve_ref(fr, st.src.parts)
            v = self.table.read_slot(seg, slot)
            self._write_cell(fr, v)
            return
        if isinstance(st, A.StoreOp):
            seg, slot = self._resolve_ref(fr, st.dst.parts)
            v = self._read_cell(fr)
            self._write_slot(seg, slot, v)
            return
        if isinstance(st, A.SwapOp):
            seg, slot = self._resolve_ref(fr, st.other.parts)
            a = self._read_cell(fr)
            b = self.table.read_slot(seg, slot)
            self._write_cell(fr, b)
            self._write_slot(seg, slot, a)
            return
        if isinstance(st, A.AssignStmt):
            rhs = self._eval_expr(st.rhs, fr)
            self._assign_ref(fr, st.lhs.parts, rhs)
            return
        if isinstance(st, A.VarDecl):
            v = self._eval_expr(st.init, fr) if st.init else 0
            fr.locals[st.name] = v
            return
        if isinstance(st, A.PtrDecl):
            p = self._eval_expr(st.init, fr)
            fr.locals[st.name] = p
            return
        if isinstance(st, A.PtrArith):
            p = fr.locals.get(st.name)
            if not isinstance(p, Pointer):
                raise BF2RuntimeError("bad ptr", st.loc)
            lay = self.structs.get(p.pointee.name) if p.pointee.name in self.structs else None
            stride = lay.size if lay else 1
            p.slot += st.delta * stride
            return
        if isinstance(st, A.PtrWrite):
            p = fr.locals.get(st.ptr)
            if not isinstance(p, Pointer):
                raise BF2RuntimeError("bad ptr", st.loc)
            v = self._eval_expr(st.value, fr)
            self._write_slot(p.seg, p.slot, v)
            return
        if isinstance(st, A.PtrRead):
            p = fr.locals.get(st.ptr)
            if not isinstance(p, Pointer):
                raise BF2RuntimeError("bad ptr", st.loc)
            self._write_cell(fr, self.table.read_slot(p.seg, p.slot))
            return
        if isinstance(st, A.IfStmt):
            v = self._read_cell(fr)
            ok = self._test_cond(st.cond, v)
            if ok:
                self._exec_block(st.then, fr)
            elif st.els:
                self._exec_block(st.els, fr)
            return
        if isinstance(st, A.LoopBF):
            while self._read_cell(fr) != 0 and not fr.returned:
                self._exec_block(st.body, fr)
            return
        if isinstance(st, A.LoopCounted):
            for _ in range(st.count):
                if fr.returned:
                    break
                self._exec_block(st.body, fr)
            return
        if isinstance(st, A.IOStmt):
            k = st.kind
            if k == ".":
                v = int(self._read_cell(fr)) & 0xFF
                self.out.append(chr(v))
            elif k == ",":
                raise BF2RuntimeError(", not in test", st.loc)
            elif k == ".i":
                if st.expr:
                    vv = self._eval_expr(st.expr, fr)
                    self.out.append(str(int(vv)))
                else:
                    self.out.append(str(int(self._read_cell(fr))))
            elif k == ".f":
                vv = self._eval_expr(st.expr, fr) if st.expr else self._read_cell(fr)
                self.out.append(str(float(vv)))
            elif k == ".s":
                self.out.append("")
            elif k == ",s":
                pass
            return
        if isinstance(st, A.CallStmt):
            self._eval_call(st.call, fr)
            return
        if isinstance(st, A.RetStmt):
            if st.value is not None:
                fr.locals["__ret"] = self._eval_expr(st.value, fr)
            fr.returned = True
            return
        if isinstance(st, A.ExprStmt):
            self._eval_expr(st.expr, fr)
            return
        if isinstance(st, A.AllocStmt):
            self._heap_seq += 1
            hid = f"__heap{self._heap_seq}"
            lay = self.structs.get(st.ty.name) if st.ty.name in self.structs else None
            decl = A.SegmentDecl(hid, st.ty, st.count, st.loc)
            self.table.add_segment(decl, lay)
            if st.name:
                fr.locals[st.name] = Pointer(hid, 0, st.ty)
            return
        if isinstance(st, A.FreeStmt):
            return
        if isinstance(st, (A.LabelStmt, A.JumpStmt)):
            return

    def _assign_ref(self, fr: Frame, parts: List[Any], val: Any) -> None:
        if not parts:
            return
        if parts[0] == "@":
            seg, slot = self._resolve_ref(fr, parts)
            self._write_slot(seg, slot, val)
            return
        head = str(parts[0])
        if head in fr.locals and isinstance(fr.locals[head], Pointer):
            p: Pointer = fr.locals[head]
            slot = p.slot
            i = 1
            lay = self.table.structs.get(p.pointee.name) if p.pointee.name in self.structs else None
            while i < len(parts):
                x = parts[i]
                if isinstance(x, str) and lay:
                    slot += lay.offsets[x]
                i += 1
            self._write_slot(p.seg, slot, val)
            return
        seg, slot = self._resolve_ref(fr, parts)
        self._write_slot(seg, slot, val)

    def _normalize_ref_indices(self, fr: Frame, parts: List[Any]) -> List[Any]:
        if not parts:
            return parts
        out: List[Any] = [parts[0]]
        for i in range(1, len(parts)):
            x = parts[i]
            if isinstance(x, A.IntLit):
                out.append(x)
            elif isinstance(x, str):
                out.append(x)
            else:
                v = self._eval_expr(x, fr)
                out.append(A.IntLit(int(v), SourceLoc(1, 1)))
        return out

    def _resolve_ref(self, fr: Frame, parts: List[Any]) -> Tuple[str, int]:
        if parts and parts[0] == "@":
            return fr.cursor_seg, fr.cursor_slot
        parts = self._normalize_ref_indices(fr, parts)
        head = str(parts[0])
        if head in fr.locals and isinstance(fr.locals[head], Pointer):
            p = fr.locals[head]
            slot = p.slot
            lay = self.structs.get(p.pointee.name) if p.pointee.name in self.structs else None
            i = 1
            while i < len(parts):
                x = parts[i]
                if isinstance(x, A.IntLit):
                    slot += int(x.value) * (lay.size if lay else 1)
                elif isinstance(x, str) and lay:
                    slot += lay.offsets[x]
                i += 1
            return p.seg, slot
        return self.table.resolve_ref(parts)

    def _read_cell(self, fr: Frame) -> Union[int, float]:
        return self.table.read_slot(fr.cursor_seg, fr.cursor_slot)

    def _write_cell(self, fr: Frame, val: Any) -> None:
        self._write_slot(fr.cursor_seg, fr.cursor_slot, val)

    def _write_slot(self, seg: str, slot: int, val: Any) -> None:
        self.table.write_slot(seg, slot, val)
        self.reactors.fire(watch_key(seg, slot), val)

    def _test_cond(self, c: A.Cond, v: Union[int, float]) -> bool:
        iv = int(v) if not isinstance(v, float) else int(v)
        if c.kind == ">0":
            return iv > 0
        if c.kind == "<0":
            return iv < 0
        if c.kind == "==0":
            return iv == 0
        if c.kind == "!=0":
            return iv != 0
        if c.kind == ">N":
            return iv > int(c.imm or 0)
        if c.kind == "<N":
            return iv < int(c.imm or 0)
        if c.kind == "==N":
            return iv == int(c.imm or 0)
        return False

    def _eval_expr(self, e: A.Expr, fr: Frame) -> Any:
        if isinstance(e, A.IntLit):
            return int(e.value)
        if isinstance(e, A.FloatLit):
            return float(e.value)
        if isinstance(e, A.BoolLit):
            return bool(e.value)
        if isinstance(e, A.Ident):
            if e.name not in fr.locals:
                raise BF2RuntimeError(f"unknown {e.name}", e.loc)
            return fr.locals[e.name]
        if isinstance(e, A.RefExpr):
            seg, slot = self._resolve_ref(fr, e.parts)
            return self.table.read_slot(seg, slot)
        if isinstance(e, A.BinOp):
            a = self._eval_expr(e.left, fr)
            b = self._eval_expr(e.right, fr)
            if e.op == "+":
                return a + b
            if e.op == "-":
                return a - b
            if e.op == "*":
                return a * b
            if e.op == "/":
                return int(a // b) if b else 0
            raise BF2RuntimeError("op", e.loc)
        if isinstance(e, A.Unary):
            if e.op == "-":
                return -self._eval_expr(e.expr, fr)
            if e.op == "*":
                if isinstance(e.expr, A.Ident):
                    p = fr.locals[e.expr.name]
                    if isinstance(p, Pointer):
                        return self.table.read_slot(p.seg, p.slot)
            if e.op == "&" and isinstance(e.expr, A.RefExpr):
                seg, slot = self._resolve_ref(fr, e.expr.parts)
                st = self.table.segments[seg]
                return Pointer(seg, slot, st.elem_type)
        if isinstance(e, A.Call):
            return self._eval_call(e, fr)
        raise BF2RuntimeError("expr", getattr(e, "loc", SourceLoc(1, 1)))

    def _eval_call(self, c: A.Call, fr: Frame) -> Any:
        if c.name == "sqrt":
            x = float(self._eval_expr(c.args[0], fr))
            return math.sqrt(x)
        if getattr(self.mod, "use_linux_stdlib", False) and c.name in (
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
            raise BF2RuntimeError(
                f"{c.name} is only available when compiling to native (libc); use bf2 compile",
                c.loc,
            )
        fn = next((x for x in self.mod.items if isinstance(x, A.FunctionDef) and x.name == c.name), None)
        if not fn:
            raise BF2RuntimeError(f"no fn {c.name}", c.loc)
        args = [self._eval_expr(a, fr) for a in c.args]
        return self.call_fn(fn, args)


def run_bf_classic(src: str) -> str:
    tape = [0] * 30000
    p = 0
    out: List[str] = []
    ip = 0
    stack: List[int] = []
    while ip < len(src):
        ch = src[ip]
        if ch in " \t\n\r":
            ip += 1
            continue
        if ch == ">":
            p += 1
            ip += 1
        elif ch == "<":
            p -= 1
            ip += 1
        elif ch == "+":
            tape[p] = (tape[p] + 1) & 0xFF
            ip += 1
        elif ch == "-":
            tape[p] = (tape[p] - 1) & 0xFF
            ip += 1
        elif ch == ".":
            out.append(chr(tape[p] & 0xFF))
            ip += 1
        elif ch == ",":
            ip += 1
        elif ch == "[":
            if tape[p] == 0:
                depth = 1
                ip += 1
                while depth and ip < len(src):
                    if src[ip] == "[":
                        depth += 1
                    elif src[ip] == "]":
                        depth -= 1
                    ip += 1
            else:
                stack.append(ip)
                ip += 1
        elif ch == "]":
            if tape[p] != 0:
                ip = stack[-1] + 1
            else:
                stack.pop()
                ip += 1
        else:
            ip += 1
    return "".join(out)


def is_classic_bf(src: str) -> bool:
    i = 0
    while i < len(src):
        c = src[i]
        if c in " \t\n\r":
            i += 1
            continue
        if c in "><+-.,[]":
            i += 1
            continue
        return False
    return True
