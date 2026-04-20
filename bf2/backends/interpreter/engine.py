from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bf2.core import ast as A
from bf2.core.errors import BF2RuntimeError, SourceLoc
from bf2.core.visitor import ASTVisitor
from bf2.backends.common.memory import Pointer, SegmentTable, StructLayout, build_struct_layout, watch_key
from bf2.backends.common.reactor import ReactorEngine


class JumpSignal(Exception):
    def __init__(self, label: str):
        self.label = label


@dataclass
class Frame:
    locals: Dict[str, Any] = field(default_factory=dict)
    cursor_seg: str = "__bf"
    cursor_slot: int = 0
    returned: bool = False


class Interpreter(ASTVisitor):
    """Interprets a Brainfhek AST using the Visitor pattern."""

    def __init__(self, mod: A.Module):
        self.mod = mod
        self.structs: Dict[str, StructLayout] = {}
        self.table: SegmentTable = SegmentTable({})
        self.reactors = ReactorEngine()
        self.out: List[str] = []
        self._heap_seq = 0
        self.current_frame: Optional[Frame] = None

    def run(self, args: Optional[List[str]] = None) -> str:
        if args is None:
            args = []
        self._ensure_bf()
        for item in self.mod.items:
            if isinstance(item, (A.StructDecl, A.SegmentDecl, A.StructStmt, A.SegmentStmt)):
                self.visit(item)
        for item in self.mod.items:
            if isinstance(item, A.ReactorDef):
                self.visit(item)
        main_fn = next((it for it in self.mod.items if isinstance(it, A.FunctionDef) and it.name == "main"), None)
        if not main_fn:
            raise BF2RuntimeError("no main() found")
        self.call_fn(main_fn, [len(args), self._create_argv(args)])
        return "".join(self.out)

    def _ensure_bf(self) -> None:
        if "__bf" not in self.table.segments:
            d = A.SegmentDecl("__bf", A.TypeRef("i8"), 30000, SourceLoc(1, 1))
            self.table.add_segment(d, None)

    def _create_argv(self, args: List[str]) -> Pointer:
        argv_pointers = []
        for i, arg in enumerate(args):
            hid = f"__arg{i}"
            data = arg.encode("utf-8") + b"\0"
            ds = A.SegmentDecl(hid, A.TypeRef("i8"), len(data), SourceLoc(1,1))
            self.table.add_segment(ds, None)
            for j, b in enumerate(data):
                self.table.write_slot(hid, j, b)
            argv_pointers.append(Pointer(hid, 0, A.TypeRef("i8")))
        ds_argv = A.SegmentDecl("__argv", A.TypeRef("ptr", A.TypeRef("i8")), len(argv_pointers), SourceLoc(1,1))
        self.table.add_segment(ds_argv, None)
        for i, ptr in enumerate(argv_pointers):
            self.table.write_slot("__argv", i, ptr)
        return Pointer("__argv", 0, A.TypeRef("ptr", A.TypeRef("i8")))

    def call_fn(self, fn: A.FunctionDef, args: List[Any]) -> Any:
        prev_frame = self.current_frame
        new_frame = Frame()
        self.current_frame = new_frame
        for (param_name, _), arg_val in zip(fn.params, args):
            new_frame.locals[param_name] = arg_val
        if fn.params:
            p0_type = fn.params[0][1]
            if p0_type.name in ("i8", "i16", "i32", "i64", "i") and isinstance(args[0], (int, float)):
                self._ensure_cp()
                self.table.write_slot("__cp", 0, int(args[0]))
                new_frame.cursor_seg, new_frame.cursor_slot = "__cp", 0
        self.visit(fn.body)
        ret_val = new_frame.locals.get("__ret", 0)
        self.current_frame = prev_frame
        return ret_val

    def _ensure_cp(self) -> None:
        if "__cp" not in self.table.segments:
            self.table.add_segment(A.SegmentDecl("__cp", A.TypeRef("i32"), 1, SourceLoc(1, 1)), None)

    def visit_block(self, node: A.Block) -> None:
        labels = {}
        for i, stmt in enumerate(node.stmts):
            if isinstance(stmt, A.LabelStmt):
                labels[stmt.name] = i

        pc = 0
        while pc < len(node.stmts):
            stmt = node.stmts[pc]
            try:
                self.visit(stmt)
                if self.current_frame.returned:
                    break
                pc += 1
            except JumpSignal as j:
                if j.label in labels:
                    pc = labels[j.label]
                else:
                    # Bubble up if label not in this block
                    raise j

    def visit_struct_decl(self, node: A.StructDecl) -> None:
        layout = build_struct_layout(node)
        self.structs[node.name] = layout
        self.table.structs[node.name] = layout

    def visit_segment_decl(self, node: A.SegmentDecl) -> None:
        layout = self.table.structs.get(node.elem_type.name)
        self.table.add_segment(node, layout)

    def visit_struct_stmt(self, node: A.StructStmt) -> None:
        self.visit(node.decl)

    def visit_segment_stmt(self, node: A.SegmentStmt) -> None:
        self.visit(node.decl)

    def visit_reactor_def(self, node: A.ReactorDef) -> None:
        seg, slot = self._resolve_ref(node.target.parts)
        key = watch_key(seg, slot)
        def callback(new_val: Any) -> None:
            prev = self.current_frame
            self.current_frame = Frame(cursor_seg=seg, cursor_slot=slot)
            self.visit(node.body)
            self.current_frame = prev
        self.reactors.register(key, callback)

    def visit_move_op(self, node: A.MoveOp) -> None:
        if len(node.target.parts) == 1 and isinstance(node.target.parts[0], str) and node.target.parts[0] in self.current_frame.locals:
            ptr = self.current_frame.locals[node.target.parts[0]]
            if isinstance(ptr, Pointer):
                self.current_frame.cursor_seg = ptr.seg
                self.current_frame.cursor_slot = ptr.slot
                return
        seg, slot = self._resolve_ref(node.target.parts)
        self.current_frame.cursor_seg, self.current_frame.cursor_slot = seg, slot

    def visit_move_rel(self, node: A.MoveRel) -> None:
        self.current_frame.cursor_slot += node.delta

    def visit_cell_arith(self, node: A.CellArith) -> None:
        val = self._read_cell()
        amount = node.amount if node.amount is not None else 1
        if node.op == "+":
            new_v = val + amount
        elif node.op == "-":
            new_v = val - amount
        elif node.op == "*":
            new_v = val * amount
        elif node.op == "/":
            new_v = int(val // amount) if amount else 0
        else:
            new_v = val
        self._write_cell(new_v)

    def visit_cell_arith_ref(self, node: A.CellArithRef) -> None:
        # Check if local
        if len(node.target.parts) == 1 and isinstance(node.target.parts[0], str) and node.target.parts[0] in self.current_frame.locals:
            name = node.target.parts[0]
            val = self.current_frame.locals[name]
            if node.op == "+":
                val += node.amount
            elif node.op == "-":
                val -= node.amount
            elif node.op == "*":
                val *= node.amount
            elif node.op == "/":
                val = int(val // node.amount) if node.amount else 0
            self.current_frame.locals[name] = val
            return
        seg, slot = self._resolve_ref(node.target.parts)
        val = self.table.read_slot(seg, slot)
        if node.op == "+":
            val += node.amount
        elif node.op == "-":
            val -= node.amount
        elif node.op == "*":
            val *= node.amount
        elif node.op == "/":
            val = int(val // node.amount) if node.amount else 0
        self._write_slot(seg, slot, val)

    def visit_cell_assign_lit(self, node: A.CellAssignLit) -> None:
        self._write_cell(node.value)

    def visit_load_op(self, node: A.LoadOp) -> None:
        seg, slot = self._resolve_ref(node.src.parts)
        self._write_cell(self.table.read_slot(seg, slot))

    def visit_store_op(self, node: A.StoreOp) -> None:
        seg, slot = self._resolve_ref(node.dst.parts)
        self._write_slot(seg, slot, self._read_cell())

    def visit_swap_op(self, node: A.SwapOp) -> None:
        seg, slot = self._resolve_ref(node.other.parts)
        tmp = self._read_cell()
        self._write_cell(self.table.read_slot(seg, slot))
        self._write_slot(seg, slot, tmp)

    def visit_assign_stmt(self, node: A.AssignStmt) -> None:
        val = self.visit(node.rhs)
        if self.current_frame and len(node.lhs.parts) == 1 and isinstance(node.lhs.parts[0], str) and node.lhs.parts[0] in self.current_frame.locals:
            self.current_frame.locals[node.lhs.parts[0]] = val
            return
        self._assign_to_ref(node.lhs.parts, val)

    def visit_var_decl(self, node: A.VarDecl) -> None:
        self.current_frame.locals[node.name] = self.visit(node.init) if node.init else 0

    def visit_ptr_decl(self, node: A.PtrDecl) -> None:
        self.current_frame.locals[node.name] = self.visit(node.init)

    def visit_ptr_arith(self, node: A.PtrArith) -> None:
        ptr = self.current_frame.locals.get(node.name)
        if not isinstance(ptr, Pointer):
            raise BF2RuntimeError(f"'{node.name}' is not a pointer")
        stride = self.table.segments[ptr.seg].struct_layout.size if self.table.segments[ptr.seg].struct_layout else 1
        ptr.slot += node.delta * stride

    def visit_ptr_write(self, node: A.PtrWrite) -> None:
        ptr = self.current_frame.locals.get(node.ptr)
        if not isinstance(ptr, Pointer):
            raise BF2RuntimeError(f"'{node.ptr}' is not a pointer")
        self._write_slot(ptr.seg, ptr.slot, self.visit(node.value))

    def visit_ptr_read(self, node: A.PtrRead) -> None:
        ptr = self.current_frame.locals.get(node.ptr)
        if not isinstance(ptr, Pointer):
            raise BF2RuntimeError(f"'{node.ptr}' is not a pointer")
        self._write_cell(self.table.read_slot(ptr.seg, ptr.slot))

    def visit_if_stmt(self, node: A.IfStmt) -> None:
        if self._test_condition(node.cond):
            self.visit(node.then)
        elif node.els:
            self.visit(node.els)

    def visit_loop_bf(self, node: A.LoopBF) -> None:
        while self._read_cell() != 0 and not self.current_frame.returned:
            self.visit(node.body)

    def visit_loop_counted(self, node: A.LoopCounted) -> None:
        for _ in range(node.count):
            if self.current_frame.returned:
                break
            self.visit(node.body)

    def visit_label_stmt(self, node: A.LabelStmt) -> None:
        pass

    def visit_jump_stmt(self, node: A.JumpStmt) -> None:
        raise JumpSignal(node.name)

    def visit_io_stmt(self, node: A.IOStmt) -> None:
        if node.kind == ".": 
            val = self.visit(node.expr) if node.expr else self._read_cell()
            self.out.append(chr(int(val) & 0xFF))
        elif node.kind == ",":
            res = ord(sys.stdin.read(1)) if not sys.stdin.isatty() else 0
            if node.expr:
                self._assign_to_ref(node.expr.parts, res)
            else:
                self._write_cell(res)
        elif node.kind.startswith("."):
            suffix = node.kind[1:]
            val = self.visit(node.expr) if node.expr else self._read_cell()
            
            # Formatted output
            out_s = ""
            if suffix in ("i", "ir", "i64", "i64r", "idx"):
                out_s = str(int(val))
                if suffix == "idx": out_s += ": "
            elif suffix in ("f", "fr"):
                out_s = str(float(val))
            elif suffix == "s":
                out_s = self._ptr_to_str(val) if isinstance(val, Pointer) else str(val)
            
            self.out.append(out_s)
            
            # Newline for i/f standard variants
            if suffix in ("i", "f", "i64"):
                self.out.append("\n")

    def visit_call_stmt(self, node: A.CallStmt) -> None:
        self.visit(node.call)

    def visit_ret_stmt(self, node: A.RetStmt) -> None:
        if node.value:
            self.current_frame.locals["__ret"] = self.visit(node.value)
        self.current_frame.returned = True

    def visit_alloc_stmt(self, node: A.AllocStmt) -> None:
        self._heap_seq += 1
        hid = f"__heap{self._heap_seq}"
        self.table.add_segment(A.SegmentDecl(hid, node.ty, node.count, node.loc), self.table.structs.get(node.ty.name))
        if node.name:
            self.current_frame.locals[node.name] = Pointer(hid, 0, node.ty)

    def visit_free_stmt(self, node: A.FreeStmt) -> None:
        pass

    def visit_expr_stmt(self, node: A.ExprStmt) -> Any:
        return self.visit(node.expr)

    # --- Expressions ---

    def visit_int_lit(self, node: A.IntLit) -> int:
        return int(node.value)

    def visit_float_lit(self, node: A.FloatLit) -> float:
        return float(node.value)

    def visit_bool_lit(self, node: A.BoolLit) -> bool:
        return bool(node.value)

    def visit_string_lit(self, node: A.StringLit) -> str:
        return str(node.value)

    def visit_ident(self, node: A.Ident) -> Any:
        if self.current_frame and node.name in self.current_frame.locals:
            return self.current_frame.locals[node.name]
        if node.name in self.table.segments:
            # Special case for raw segment access in expressions
            return Pointer(node.name, 0, self.table.segments[node.name].elem_type)
        raise BF2RuntimeError(f"unknown identifier: {node.name}")

    def visit_ref_expr(self, node: A.RefExpr) -> Any:
        # Check if local
        if len(node.parts) == 1 and isinstance(node.parts[0], str) and node.parts[0] in self.current_frame.locals:
            return self.current_frame.locals[node.parts[0]]
        seg, slot = self._resolve_ref(node.parts)
        return self.table.read_slot(seg, slot)

    def visit_bin_op(self, node: A.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        if node.op == "+":
            if isinstance(left, Pointer) and isinstance(right, int):
                stride = self.table.segments[left.seg].struct_layout.size if self.table.segments[left.seg].struct_layout else 1
                return Pointer(left.seg, left.slot + right * stride, left.pointee)
            if isinstance(right, Pointer) and isinstance(left, int):
                stride = self.table.segments[right.seg].struct_layout.size if self.table.segments[right.seg].struct_layout else 1
                return Pointer(right.seg, right.slot + left * stride, right.pointee)
            return left + right
        if node.op == "-":
            if isinstance(left, Pointer) and isinstance(right, int):
                stride = self.table.segments[left.seg].struct_layout.size if self.table.segments[left.seg].struct_layout else 1
                return Pointer(left.seg, left.slot - right * stride, left.pointee)
            if isinstance(left, Pointer) and isinstance(right, Pointer):
                return left.slot - right.slot
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return int(left // right) if right else 0
        if node.op == "==":
            return left == right
        if node.op == "!=":
            return left != right
        if node.op == "<":
            return left < right
        if node.op == ">":
            return left > right
        if node.op == "<=":
            return left <= right
        if node.op == ">=":
            return left >= right
        raise BF2RuntimeError(f"unsupported binop: {node.op}")

    def visit_unary(self, node: A.Unary) -> Any:
        v = self.visit(node.expr)
        if node.op == "-":
            return -v
        if node.op == "*":
            if isinstance(v, Pointer):
                return self.table.read_slot(v.seg, v.slot)
            raise BF2RuntimeError(f"dereferencing non-pointer: {v}")
        if node.op == "&":
            if isinstance(node.expr, A.RefExpr):
                seg, slot = self._resolve_ref(node.expr.parts)
                return Pointer(seg, slot, self.table.segments[seg].elem_type)
        raise BF2RuntimeError(f"unsupported unary: {node.op}")

    def visit_call(self, node: A.Call) -> Any:
        if node.name == "sqrt":
            return math.sqrt(float(self.visit(node.args[0])))
        if node.name == "open":
            f_ptr = self.visit(node.args[0])
            return os.open(self._ptr_to_str(f_ptr), os.O_RDONLY)
        if node.name == "read":
            fd, b_ptr, cnt = self.visit(node.args[0]), self.visit(node.args[1]), self.visit(node.args[2])
            data = os.read(fd, cnt)
            for i, b in enumerate(data):
                self.table.write_slot(b_ptr.seg, b_ptr.slot + i, b)
            return len(data)
        if node.name == "close":
            os.close(self.visit(node.args[0]))
            return 0
        fn = next((it for it in self.mod.items if isinstance(it, A.FunctionDef) and it.name == node.name), None)
        if not fn:
            raise BF2RuntimeError(f"unknown function: {node.name}")
        return self.call_fn(fn, [self.visit(a) for a in node.args])

    # --- Helpers ---

    def _ptr_to_str(self, ptr: Pointer) -> str:
        res, i = [], 0
        while True:
            c = self.table.read_slot(ptr.seg, ptr.slot + i)
            if c == 0:
                break
            res.append(chr(int(c)))
            i += 1
        return "".join(res)

    def _read_cell(self) -> Any:
        return self.table.read_slot(self.current_frame.cursor_seg, self.current_frame.cursor_slot)

    def _write_cell(self, val: Any) -> None:
        self._write_slot(self.current_frame.cursor_seg, self.current_frame.cursor_slot, val)

    def _write_slot(self, seg: str, slot: int, val: Any) -> None:
        self.table.write_slot(seg, slot, val)
        self.reactors.fire(watch_key(seg, slot), val)

    def _resolve_ref(self, parts: List[Any]) -> Tuple[str, int]:
        if self.current_frame and parts and parts[0] == "@":
            return self.current_frame.cursor_seg, self.current_frame.cursor_slot
        head = str(parts[0])
        if self.current_frame and head in self.current_frame.locals and isinstance(self.current_frame.locals[head], Pointer):
            ptr = self.current_frame.locals[head]
            seg, slot, lay = ptr.seg, ptr.slot, self.table.segments[ptr.seg].struct_layout
            for p in parts[1:]:
                if isinstance(p, A.IntLit):
                    slot += int(p.value) * (lay.size if lay else 1)
                elif isinstance(p, A.Expr):
                    slot += int(self.visit(p)) * (lay.size if lay else 1)
                elif isinstance(p, str) and lay:
                    slot += lay.offsets[p]
            return seg, slot
        norm_parts = [parts[0]]
        for p in parts[1:]:
            if isinstance(p, A.Expr):
                norm_parts.append(A.IntLit(int(self.visit(p)), SourceLoc(1,1)))
            else:
                norm_parts.append(p)
        return self.table.resolve_ref(norm_parts)

    def _assign_to_ref(self, parts: List[Any], val: Any) -> None:
        seg, slot = self._resolve_ref(parts)
        self._write_slot(seg, slot, val)

    def _test_condition(self, cond: A.Cond) -> bool:
        v = int(self.visit(cond.expr)) if cond.expr else int(self._read_cell())
        imm = int(cond.imm or 0)
        if cond.kind == ">0":
            return v > 0
        if cond.kind == "<0":
            return v < 0
        if cond.kind == "==0":
            return v == 0
        if cond.kind == "!=0":
            return v != 0
        if cond.kind == ">N":
            return v > imm
        if cond.kind == "<N":
            return v < imm
        if cond.kind == "==N":
            return v == imm
        if cond.kind == "!=N":
            return v != imm
        return False

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

def run_bf_classic(src: str) -> str:
    tape = [0] * 30000
    p = 0
    out = []
    stack = []
    ip = 0
    while ip < len(src):
        ch = src[ip]
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
