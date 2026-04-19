from __future__ import annotations

from typing import Any, Dict, List, Tuple
import platform

from bf2.core import ast as A
from bf2.core.errors import BF2Error
from bf2.backends.llvm.context import LLVMContext

class LLVMGenError(BF2Error):
    """Raised when LLVM IR generation fails."""

def _scalar_ty(t: A.TypeRef) -> str:
    if t.name == "ptr": return "ptr"
    if t.name == "i8": return "i8"
    if t.name == "bool": return "i1"
    if t.name == "i16": return "i16"
    if t.name == "i32": return "i32"
    if t.name == "i64": return "i64"
    if t.name == "f32": return "float"
    if t.name == "f64": return "double"
    return f"%struct.{t.name}" if t.name not in ("void", "") else "void"

def _align(t: str) -> int:
    if t == "i8": return 1
    if t == "i1" or t == "bool": return 1
    if t == "i16": return 2
    if t == "i32" or t == "float": return 4
    if t == "i64" or t == "double" or t == "ptr": return 8
    return 8

_LINUX_LIBC = {
    "write": ("i64", ["i32", "ptr", "i64"]),
    "read": ("i64", ["i32", "ptr", "i64"]),
    "open": ("i32", ["ptr", "i32"]),
    "close": ("i32", ["i32"]),
    "exit": ("void", ["i32"]),
    "fork": ("i32", []),
    "getpid": ("i32", []),
    "strlen": ("i64", ["ptr"]),
    "nanosleep": ("i32", ["ptr", "ptr"]),
    "printf": ("i32", ["ptr", "..."]),
}

class LLVMEmitterVisitor:
    def __init__(self, mod: A.Module, target: str):
        self.mod = mod
        self.target = target
        self.ctx = LLVMContext("void")
        self.lines: List[str] = []
        self.structs: Dict[str, A.StructDecl] = {}
        self.fns: Dict[str, A.FunctionDef] = {}
        self.global_segs: Dict[str, A.SegmentDecl] = {}
        self.seg_slots: Dict[str, str] = {} # seg -> global ptr
        self.watches: List[Tuple[str, int, A.ReactorDef]] = []

    def emit(self) -> str:
        # Preamble
        self.lines.append(f'target triple = "{self.target}"')
        
        use_linux = getattr(self.mod, "use_linux_stdlib", False)
        if not use_linux:
            self.lines.append('declare i32 @putchar(i32)')
            self.lines.append('declare i32 @getchar()')
            self.lines.append('declare i32 @scanf(ptr, ...)')
            self.lines.append('declare i32 @printf(ptr, ...)')
            self.lines.append('declare i64 @strlen(ptr)')
            self.lines.append('declare ptr @malloc(i64)')
            self.lines.append('declare void @free(ptr)')
        
        if use_linux:
            for name, (ret, args) in _LINUX_LIBC.items():
                if name == "printf": continue # Redundant
                arg_s = ", ".join(args)
                self.lines.append(f"declare {ret} @{name}({arg_s})")

        # Built-in segments for cursor
        self.lines.append("@__bf = global [30000 x i8] zeroinitializer, align 16")
        self.lines.append("@__cseg = global ptr null, align 8")
        self.lines.append("@__cslot = global i32 0, align 4")
        self.lines.append("@__.fmt_i = private unnamed_addr constant [4 x i8] c\"%d\\0A\\00\", align 1")
        self.lines.append("@__.fmt_f = private unnamed_addr constant [7 x i8] c\"%.17g\\0A\\00\", align 1")
        self.lines.append("@__.fmt_s = private unnamed_addr constant [4 x i8] c\"%s\\00\\00\", align 1")
        self.lines.append("@bf2.watch.depth = global i32 0, align 4")
        
        if getattr(self.mod, "use_linux_stdlib", False):
            self.lines.append("%struct.timespec = type { i64, i64 }")
            self.lines.append("@STDOUT = global i32 1, align 4")
            self.lines.append("@O_CREAT = global i32 64, align 4")
            self.lines.append("declare i32 @snprintf(ptr, i64, ptr, ...)")

        # Collect top-levels
        for item in self.mod.items:
            if isinstance(item, A.StructDecl): self.structs[item.name] = item
            elif isinstance(item, A.FunctionDef): self.fns[item.name] = item
            elif isinstance(item, A.SegmentDecl): self.global_segs[item.name] = item
            elif isinstance(item, A.ReactorDef):
                sk = self._try_static_seg_slot(item.target)
                if sk:
                    self.watches.append((sk[0], sk[1], item))

        # Emit Structs
        for s in self.structs.values():
            fields = ", ".join(_scalar_ty(t) for _, t in s.fields)
            self.lines.append(f"%struct.{s.name} = type {{ {fields} }}")

        # Emit Global Segments
        for s in self.global_segs.values():
            lty = _scalar_ty(s.elem_type)
            sz = s.length
            self.lines.append(f"@{s.name} = global [{sz} x {lty}] zeroinitializer, align {_align(lty)}")
            self.seg_slots[s.name] = f"@{s.name}"

        # Emit Functions
        for f in self.fns.values(): self._emit_function(f)

        # Emit Reactors
        for i, (seg, slot, r) in enumerate(self.watches):
            self._emit_watch_fn(i, seg, slot, r)

        return "\n".join(self.lines)

    def _emit_watch_fn(self, idx: int, seg: str, slot: int, r: A.ReactorDef):
        self.ctx = LLVMContext("void")
        self.lines.append(f"define void @bf2.watch.{idx}() {{")
        self.lines.append("entry:")
        # Reactors run with cursor at watched slot
        p = self.seg_slots[seg]
        self.lines.append(f"  store ptr {p}, ptr @__cseg, align 8")
        self.lines.append(f"  store i32 {slot}, ptr @__cslot, align 4")
        self.ctx.cursor_type = _scalar_ty(self.global_segs[seg].elem_type)
        for stmt in r.body.stmts: self._emit_stmt(stmt)
        self.lines.append("  ret void")
        self.lines.append("}")

    def _emit_function(self, f: A.FunctionDef):
        ret_ty = _scalar_ty(f.ret)
        self.ctx = LLVMContext(ret_ty)
        args = ", ".join(f"{_scalar_ty(t)} %{n}" for n, t in f.params)
        self.lines.append(f"define {ret_ty} @{f.name}({args}) {{")
        self.lines.append("entry:")
        
        # Alloc parameters
        for n, t in f.params:
            lty = _scalar_ty(t)
            p = "%" + self.ctx.next_temp(f"p.{n}")
            self.lines.append(f"  {p} = alloca {lty}, align {_align(lty)}")
            self.lines.append(f"  store {lty} %{n}, ptr {p}, align {_align(lty)}")
            self.ctx.locals[n] = (p, lty)
            if t.name == "ptr": self.ctx.ptr_inner[n] = t.inner or A.TypeRef("i8")
            
        if f.params:
            first_n = f.params[0][0]
            first_p, _ = self.ctx.locals[first_n]
            self.lines.append(f"  store ptr {first_p}, ptr @__cseg, align 8")
            self.lines.append(f"  store i32 0, ptr @__cslot, align 4")
            self.ctx.cursor_type = _scalar_ty(f.params[0][1])
        else:
            self.lines.append(f"  store ptr @__bf, ptr @__cseg, align 8")
            self.lines.append(f"  store i32 0, ptr @__cslot, align 4")
            self.ctx.cursor_type = "i8"

        for stmt in f.body.stmts: self._emit_stmt(stmt)
        
        if ret_ty == "void": self.lines.append("  ret void")
        elif not self.lines[-1].strip().startswith("ret "):
             self.lines.append(f"  ret {ret_ty} 0")
        
        self.lines.append("}")

    def _emit_stmt(self, stmt: A.ASTNode):
        if isinstance(stmt, A.SegmentStmt): self._emit_local_seg(stmt.decl)
        elif isinstance(stmt, A.VarDecl): self._emit_var_decl(stmt)
        elif isinstance(stmt, A.PtrDecl): self._emit_ptr_decl(stmt)
        elif isinstance(stmt, A.AssignStmt): self._emit_assign(stmt)
        elif isinstance(stmt, A.IfStmt): self._emit_if(stmt)
        elif isinstance(stmt, A.LoopBF): self._emit_loop_bf(stmt)
        elif isinstance(stmt, A.LoopCounted): self._emit_loop_counted(stmt)
        elif isinstance(stmt, A.RetStmt): self._emit_ret(stmt)
        elif isinstance(stmt, A.CallStmt): self._emit_expr(stmt.call, self.ctx)
        elif isinstance(stmt, A.MoveOp): self._emit_move_op(stmt)
        elif isinstance(stmt, A.MoveRel): self._emit_move_rel(stmt)
        elif isinstance(stmt, A.CellArith): self._emit_cell_arith(stmt)
        elif isinstance(stmt, A.CellArithRef): self._emit_cell_arith_ref(stmt)
        elif isinstance(stmt, A.CellAssignLit): self._emit_cell_assign_lit(stmt)
        elif isinstance(stmt, A.IOStmt): self._emit_io(stmt)
        elif isinstance(stmt, A.LabelStmt): self.lines.append(f"{stmt.name}:")
        elif isinstance(stmt, A.JumpStmt): self.lines.append(f"  br label %{stmt.name}")
        elif isinstance(stmt, A.PtrArith): self._emit_ptr_arith_stmt(stmt)
        elif isinstance(stmt, A.PtrRead): self._emit_ptr_read(stmt.ptr, stmt.loc)
        elif isinstance(stmt, A.PtrWrite): self._emit_ptr_write(stmt)
        elif isinstance(stmt, A.AllocStmt): self._emit_alloc(stmt)
        elif isinstance(stmt, A.FreeStmt): self._emit_free(stmt.ptr)
        elif isinstance(stmt, A.RefExpr): self._emit_expr(stmt, self.ctx)
        elif isinstance(stmt, A.ExprStmt): self._emit_expr(stmt.expr, self.ctx)

    def _try_static_seg_slot(self, r: A.RefExpr) -> Optional[Tuple[str, int]]:
        if len(r.parts) >= 1:
            head = r.parts[0]
            if head == "@":
                seg_name = r.parts[1]
                slot = 0
                return (seg_name, slot)
            if head in self.global_segs:
                seg_name = head
                slot = 0
                if len(r.parts) > 1 and isinstance(r.parts[1], A.IntLit):
                    slot = r.parts[1].value
                return (seg_name, slot)
        return None

    def _emit_maybe_watch(self, seg: str, slot_ssa: str):
        # Static slot watching only.
        for i, (wseg, wslot, _) in enumerate(self.watches):
            if wseg == seg:
                l_skip = self.ctx.next_temp("skip.watch")
                l_fire = self.ctx.next_temp("fire.watch")
                l_join = self.ctx.next_temp("join.watch")
                
                # Compare slot
                is_slot = "%" + self.ctx.next_temp("is_slot")
                # slot_ssa is a value, wslot is an int.
                # We need to make sure we're comparing correctly.
                # If slot_ssa is a register name, use it.
                self.lines.append(f"  {is_slot} = icmp eq i32 {slot_ssa}, {wslot}")
                
                # Depth check
                d = "%" + self.ctx.next_temp("depth")
                self.lines.append(f"  {d} = load i32, ptr @bf2.watch.depth, align 4")
                can_fire = "%" + self.ctx.next_temp("can_fire")
                self.lines.append(f"  {can_fire} = icmp slt i32 {d}, 8") # recursion limit
                
                must_fire = "%" + self.ctx.next_temp("must_fire")
                self.lines.append(f"  {must_fire} = and i1 {is_slot}, {can_fire}")
                
                self.lines.append(f"  br i1 {must_fire}, label %{l_fire}, label %{l_skip}")
                self.lines.append(f"{l_fire}:")
                nd = "%" + self.ctx.next_temp("ndepth")
                self.lines.append(f"  {nd} = add i32 {d}, 1")
                self.lines.append(f"  store i32 {nd}, ptr @bf2.watch.depth, align 4")
                self.lines.append(f"  call void @bf2.watch.{i}()")
                # Restore cursor after reactor
                # (Reactor might have moved it)
                # This is tricky in LLVM without a stack of cursors,
                # but BF2 reactors are supposed to be "transparent".
                # The old code didn't restore it, but let's see.
                self.lines.append(f"  store i32 {d}, ptr @bf2.watch.depth, align 4")
                self.lines.append(f"  br label %{l_join}")
                self.lines.append(f"{l_skip}:")
                self.lines.append(f"  br label %{l_join}")
                self.lines.append(f"{l_join}:")

    def _emit_maybe_watch_current(self):
        # Current cell is at @__cseg, @__cslot
        # We need to load @__cseg and compare with watches
        seg_ptr = "%" + self.ctx.next_temp("cseg_ptr")
        self.lines.append(f"  {seg_ptr} = load ptr, ptr @__cseg, align 8")
        slot_v = "%" + self.ctx.next_temp("cslot_val")
        self.lines.append(f"  {slot_v} = load i32, ptr @__cslot, align 4")
        
        for i, (wseg, wslot, _) in enumerate(self.watches):
            l_skip = self.ctx.next_temp("skip.watch")
            l_fire = self.ctx.next_temp("fire.watch")
            l_join = self.ctx.next_temp("join.watch")
            
            wseg_ptr = self.seg_slots[wseg]
            is_seg = "%" + self.ctx.next_temp("is_seg")
            self.lines.append(f"  {is_seg} = icmp eq ptr {seg_ptr}, {wseg_ptr}")
            is_slot = "%" + self.ctx.next_temp("is_slot")
            self.lines.append(f"  {is_slot} = icmp eq i32 {slot_v}, {wslot}")
            match = "%" + self.ctx.next_temp("match")
            self.lines.append(f"  {match} = and i1 {is_seg}, {is_slot}")
            
            d = "%" + self.ctx.next_temp("depth")
            self.lines.append(f"  {d} = load i32, ptr @bf2.watch.depth, align 4")
            can_fire = "%" + self.ctx.next_temp("can_fire")
            self.lines.append(f"  {can_fire} = icmp slt i32 {d}, 8")
            
            must_fire = "%" + self.ctx.next_temp("must_fire")
            self.lines.append(f"  {must_fire} = and i1 {match}, {can_fire}")
            
            self.lines.append(f"  br i1 {must_fire}, label %{l_fire}, label %{l_skip}")
            self.lines.append(f"{l_fire}:")
            nd = "%" + self.ctx.next_temp("ndepth")
            self.lines.append(f"  {nd} = add i32 {d}, 1")
            self.lines.append(f"  store i32 {nd}, ptr @bf2.watch.depth, align 4")
            self.lines.append(f"  call void @bf2.watch.{i}()")
            # Restore cursor??
            self.lines.append(f"  store ptr {seg_ptr}, ptr @__cseg, align 8")
            self.lines.append(f"  store i32 {slot_v}, ptr @__cslot, align 4")
            self.lines.append(f"  store i32 {d}, ptr @bf2.watch.depth, align 4")
            self.lines.append(f"  br label %{l_join}")
            self.lines.append(f"{l_skip}:")
            self.lines.append(f"  br label %{l_join}")
            self.lines.append(f"{l_join}:")

    def _emit_local_seg(self, d: A.SegmentDecl):
        lty = _scalar_ty(d.elem_type)
        p = "%" + self.ctx.next_temp(f"lseg.{d.name}")
        self.lines.append(f"  {p} = alloca [{d.length} x {lty}], align {_align(lty)}")
        self.ctx.locals[d.name] = (p, f"[{d.length} x {lty}]")
        self.seg_slots[d.name] = p

    def _emit_var_decl(self, d: A.VarDecl):
        lty = _scalar_ty(d.ty)
        p = "%" + self.ctx.next_temp(f"v.{d.name}")
        self.lines.append(f"  {p} = alloca {lty}, align {_align(lty)}")
        self.ctx.locals[d.name] = (p, lty)
        if d.init:
            rv, rt = self._emit_expr(d.init, self.ctx)
            # Basic type conversion for numbers
            if lty == "double" and rt == "i32":
                c = "%" + self.ctx.next_temp("conv")
                self.lines.append(f"  {c} = sitofp i32 {rv} to double")
                rv = c
            elif lty == "i32" and rt == "i8":
                c = "%" + self.ctx.next_temp("zext")
                self.lines.append(f"  {c} = zext i8 {rv} to i32")
                rv = c
            elif lty == "i32" and rt == "i64":
                c = "%" + self.ctx.next_temp("trunc")
                self.lines.append(f"  {c} = trunc i64 {rv} to i32")
                rv = c
            elif lty == "i64" and rt == "i32":
                c = "%" + self.ctx.next_temp("zext")
                self.lines.append(f"  {c} = zext i32 {rv} to i64")
                rv = c
            self.lines.append(f"  store {lty} {rv}, ptr {p}, align {_align(lty)}")

    def _emit_ptr_decl(self, d: A.PtrDecl):
        p = "%" + self.ctx.next_temp(f"ptr.{d.name}")
        self.lines.append(f"  {p} = alloca ptr, align 8")
        self.ctx.locals[d.name] = (p, "ptr")
        self.ctx.ptr_inner[d.name] = d.inner
        if d.init:
            rv, _ = self._emit_expr(d.init, self.ctx)
            self.lines.append(f"  store ptr {rv}, ptr {p}, align 8")

    def _emit_assign(self, s: A.AssignStmt):
        ptr, ty = self._gep(s.lhs, self.ctx)
        rv, rt = self._emit_expr(s.rhs, self.ctx)
        if ty == "double" and rt == "i32":
            c = "%" + self.ctx.next_temp("conv")
            self.lines.append(f"  {c} = sitofp i32 {rv} to double")
            rv = c
        elif ty == "i32" and rt == "i8":
            c = "%" + self.ctx.next_temp("zext")
            self.lines.append(f"  {c} = zext i8 {rv} to i32")
            rv = c
        elif ty == "i32" and rt == "i64":
            c = "%" + self.ctx.next_temp("trunc")
            self.lines.append(f"  {c} = trunc i64 {rv} to i32")
            rv = c
        elif ty == "i64" and rt == "i32":
            c = "%" + self.ctx.next_temp("zext")
            self.lines.append(f"  {c} = zext i32 {rv} to i64")
            rv = c
        self.lines.append(f"  store {ty} {rv}, ptr {ptr}, align {_align(ty)}")
        sk = self._try_static_seg_slot(s.lhs)
        if sk:
            self._emit_maybe_watch(sk[0], str(sk[1]))

    def _emit_if(self, s: A.IfStmt):
        cond_v = self._emit_cond(s.cond)
        l_then = self.ctx.next_temp("if.then")
        l_else = self.ctx.next_temp("if.else")
        l_end = self.ctx.next_temp("if.end")
        
        self.lines.append(f"  br i1 {cond_v}, label %{l_then}, label %{l_else}")
        self.lines.append(f"{l_then}:")
        for st in s.then.stmts: self._emit_stmt(st)
        self.lines.append(f"  br label %{l_end}")
        self.lines.append(f"{l_else}:")
        if s.els:
            for st in s.els.stmts: self._emit_stmt(st)
        self.lines.append(f"  br label %{l_end}")
        self.lines.append(f"{l_end}:")

    def _emit_loop_bf(self, s: A.LoopBF):
        l_head = self.ctx.next_temp("loop.head")
        l_body = self.ctx.next_temp("loop.body")
        l_end = self.ctx.next_temp("loop.end")
        
        self.lines.append(f"  br label %{l_head}")
        self.lines.append(f"{l_head}:")
        # Loop while current cell != 0
        p, ty = self._get_current_cell_ptr()
        t = "%" + self.ctx.next_temp("cv")
        self.lines.append(f"  {t} = load {ty}, ptr {p}, align {_align(ty)}")
        is_zero = "%" + self.ctx.next_temp("iszero")
        if ty == "double": self.lines.append(f"  {is_zero} = fcmp une double {t}, 0.0")
        else: self.lines.append(f"  {is_zero} = icmp ne {ty} {t}, 0")
        self.lines.append(f"  br i1 {is_zero}, label %{l_body}, label %{l_end}")
        
        self.lines.append(f"{l_body}:")
        for st in s.body.stmts: self._emit_stmt(st)
        self.lines.append(f"  br label %{l_head}")
        self.lines.append(f"{l_end}:")

    def _emit_loop_counted(self, s: A.LoopCounted):
        l_head = self.ctx.next_temp("loopc.head")
        l_body = self.ctx.next_temp("loopc.body")
        l_end = self.ctx.next_temp("loopc.end")
        i_ptr = "%" + self.ctx.next_temp("loopc.i")
        self.lines.append(f"  {i_ptr} = alloca i32, align 4")
        self.lines.append(f"  store i32 0, ptr {i_ptr}, align 4")
        self.lines.append(f"  br label %{l_head}")
        self.lines.append(f"{l_head}:")
        iv = "%" + self.ctx.next_temp("iv")
        self.lines.append(f"  {iv} = load i32, ptr {i_ptr}, align 4")
        cond = "%" + self.ctx.next_temp("icond")
        self.lines.append(f"  {cond} = icmp slt i32 {iv}, {s.count}")
        self.lines.append(f"  br i1 {cond}, label %{l_body}, label %{l_end}")
        self.lines.append(f"{l_body}:")
        for st in s.body.stmts: self._emit_stmt(st)
        nv = "%" + self.ctx.next_temp("ne")
        self.lines.append(f"  {nv} = add nsw i32 {iv}, 1")
        self.lines.append(f"  store i32 {nv}, ptr {i_ptr}, align 4")
        self.lines.append(f"  br label %{l_head}")
        self.lines.append(f"{l_end}:")

    def _emit_ret(self, s: A.RetStmt):
        if not s.value: self.lines.append("  ret void")
        else:
            rv, rt = self._emit_expr(s.value, self.ctx)

            self.lines.append(f"  ret {rt} {rv}")

    def _emit_move_op(self, s: A.MoveOp):
        ptr, ty = self._gep(s.target, self.ctx)
        self.ctx.cursor_type = ty
        self.lines.append(f"  store ptr {ptr}, ptr @__cseg, align 8")
        self.lines.append("  store i32 0, ptr @__cslot, align 4")

    def _emit_move_rel(self, s: A.MoveRel):
        cv = "%" + self.ctx.next_temp("idx")
        self.lines.append(f"  {cv} = load i32, ptr @__cslot, align 4")
        nv = "%" + self.ctx.next_temp("nidx")
        self.lines.append(f"  {nv} = add nsw i32 {cv}, {s.delta}")
        self.lines.append(f"  store i32 {nv}, ptr @__cslot, align 4")

    def _emit_cell_arith(self, s: A.CellArith):
        p, ty = self._get_current_cell_ptr()
        v = "%" + self.ctx.next_temp("val")
        self.lines.append(f"  {v} = load {ty}, ptr {p}, align {_align(ty)}")
        amount = s.amount if s.amount is not None else 1
        nv = "%" + self.ctx.next_temp("nv")
        if ty == "double":
            op = {"+":"fadd","-":"fsub","*":"fmul","/":"fdiv"}[s.op]
            self.lines.append(f"  {nv} = {op} double {v}, {float(amount)}")
        else:
            op = {"+":"add","-":"sub","*":"mul","/":"sdiv"}[s.op]
            self.lines.append(f"  {nv} = {op} nsw {ty} {v}, {int(amount)}")
        self.lines.append(f"  store {ty} {nv}, ptr {p}, align {_align(ty)}")

    def _emit_cell_arith_ref(self, s: A.CellArithRef):
        p, ty = self._gep(s.target, self.ctx)
        v = "%" + self.ctx.next_temp("val")
        self.lines.append(f"  {v} = load {ty}, ptr {p}, align {_align(ty)}")
        amount = s.amount if s.amount is not None else 1
        nv = "%" + self.ctx.next_temp("nv")
        if ty == "double":
            op = {"+":"fadd","-":"fsub","*":"fmul","/":"fdiv"}[s.op]
            self.lines.append(f"  {nv} = {op} double {v}, {float(amount)}")
        else:
            op = {"+":"add","-":"sub","*":"mul","/":"sdiv"}[s.op]
            self.lines.append(f"  {nv} = {op} nsw {ty} {v}, {int(amount)}")
        self.lines.append(f"  store {ty} {nv}, ptr {p}, align {_align(ty)}")
        sk = self._try_static_seg_slot(s.target)
        if sk: self._emit_maybe_watch(sk[0], str(sk[1]))

    def _emit_cell_assign_lit(self, s: A.CellAssignLit):
        p, ty = self._get_current_cell_ptr()
        if ty == "double": self.lines.append(f"  store double {float(s.value)}, ptr {p}, align 8")
        elif ty == "i1": self.lines.append(f"  store i1 {'1' if s.value else '0'}, ptr {p}, align 1")
        else: self.lines.append(f"  store {ty} {int(s.value)}, ptr {p}, align {_align(ty)}")
        self._emit_maybe_watch_current()

    def _emit_load_op(self, s: A.LoadOp):
        sp, sty = self._gep(s.src, self.ctx)
        v = "%" + self.ctx.next_temp("lv")
        self.lines.append(f"  {v} = load {sty}, ptr {sp}, align {_align(sty)}")
        dp, dty = self._get_current_cell_ptr()
        # Conversion if needed
        self.lines.append(f"  store {sty} {v}, ptr {dp}, align {_align(sty)}")
        self._emit_maybe_watch_current()

    def _emit_store_op(self, s: A.StoreOp):
        sp, sty = self._get_current_cell_ptr()
        v = "%" + self.ctx.next_temp("sv")
        self.lines.append(f"  {v} = load {sty}, ptr {sp}, align {_align(sty)}")
        dp, dty = self._gep(s.dst, self.ctx)
        self.lines.append(f"  store {sty} {v}, ptr {dp}, align {_align(sty)}")
        sk = self._try_static_seg_slot(s.dst)
        if sk: self._emit_maybe_watch(sk[0], str(sk[1]))

    def _emit_swap_op(self, s: A.SwapOp):
        p0, t0 = self._get_current_cell_ptr()
        v0 = "%" + self.ctx.next_temp("v0")
        self.lines.append(f"  {v0} = load {t0}, ptr {p0}, align {_align(t0)}")
        p1, t1 = self._gep(s.other, self.ctx)
        v1 = "%" + self.ctx.next_temp("v1")
        self.lines.append(f"  {v1} = load {t1}, ptr {p1}, align {_align(t1)}")
        self.lines.append(f"  store {t1} {v1}, ptr {p0}, align {_align(t1)}")
        self.lines.append(f"  store {t0} {v0}, ptr {p1}, align {_align(t0)}")
        self._emit_maybe_watch_current()
        sk = self._try_static_seg_slot(s.other)
        if sk: self._emit_maybe_watch(sk[0], str(sk[1]))

    def _emit_io(self, s: A.IOStmt):
        use_linux = getattr(self.mod, "use_linux_stdlib", False)
        if s.kind == ".":
            p, ty = self._get_current_cell_ptr()
            v = "%" + self.ctx.next_temp("ov")
            self.lines.append(f"  {v} = load {ty}, ptr {p}, align {_align(ty)}")
            if use_linux:
                b = "%" + self.ctx.next_temp("buf")
                self.lines.append(f"  {b} = alloca i8, align 1")
                if ty == "i8":
                    self.lines.append(f"  store i8 {v}, ptr {b}, align 1")
                else:
                    tv = "%" + self.ctx.next_temp("trunc")
                    self.lines.append(f"  {tv} = trunc {ty} {v} to i8")
                    self.lines.append(f"  store i8 {tv}, ptr {b}, align 1")
                self.lines.append(f"  call i64 @write(i32 1, ptr {b}, i64 1)")
            else:
                ev = "%" + self.ctx.next_temp("ext")
                if ty == "i32":
                    self.lines.append(f"  call i32 @putchar(i32 {v})")
                else:
                    self.lines.append(f"  {ev} = zext {ty} {v} to i32")
                    self.lines.append(f"  call i32 @putchar(i32 {ev})")
        elif s.kind == ".i" and s.expr:
            v, vty = self._emit_expr(s.expr, self.ctx)
            if use_linux:
                self._emit_snprintf_write(v, vty, "@__.fmt_i")
            else:
                self.lines.append(f"  call i32 (ptr, ...) @printf(ptr @__.fmt_i, {vty} {v})")
        elif s.kind == ".i":
            p, ty = self._get_current_cell_ptr()
            v = "%" + self.ctx.next_temp("ov")
            self.lines.append(f"  {v} = load {ty} , ptr {p}, align {_align(ty)}")
            if use_linux:
                self._emit_snprintf_write(v, ty, "@__.fmt_i")
            else:
                self.lines.append(f"  call i32 (ptr, ...) @printf(ptr @__.fmt_i, {ty} {v})")
        elif s.kind == ".f" and s.expr:
            v, vty = self._emit_expr(s.expr, self.ctx)
            if use_linux:
                self._emit_snprintf_write(v, vty, "@__.fmt_f")
            else:
                self.lines.append(f"  call i32 (ptr, ...) @printf(ptr @__.fmt_f, {vty} {v})")
        elif s.kind == ".f":
            p, ty = self._get_current_cell_ptr()
            v = "%" + self.ctx.next_temp("ov")
            self.lines.append(f"  {v} = load {ty}, ptr {p}, align {_align(ty)}")
            if use_linux:
                self._emit_snprintf_write(v, ty, "@__.fmt_f")
            else:
                self.lines.append(f"  call i32 (ptr, ...) @printf(ptr @__.fmt_f, {ty} {v})")
        elif s.kind == ".s":
            p, ty = self._get_current_cell_ptr()
            v = "%" + self.ctx.next_temp("ov")
            self.lines.append(f"  {v} = load {ty}, ptr {p}, align {_align(ty)}")
            if use_linux:
                ln = "%" + self.ctx.next_temp("len")
                self.lines.append(f"  {ln} = call i64 @strlen(ptr {v})")
                self.lines.append(f"  call i64 @write(i32 1, ptr {v}, i64 {ln})")
            else:
                self.lines.append(f"  call i32 (ptr, ...) @printf(ptr @__.fmt_s, {ty} {v})")
        elif s.kind == ",":
            p, ty = self._get_current_cell_ptr()
            if use_linux:
                b = "%" + self.ctx.next_temp("buf")
                self.lines.append(f"  {b} = alloca i8, align 1")
                self.lines.append(f"  call i64 @read(i32 0, ptr {b}, i64 1)")
                rv = "%" + self.ctx.next_temp("rv")
                self.lines.append(f"  {rv} = load i8, ptr {b}, align 1")
                if ty == "i8":
                    self.lines.append(f"  store i8 {rv}, ptr {p}, align 1")
                else:
                    tv = "%" + self.ctx.next_temp("ext")
                    self.lines.append(f"  {tv} = zext i8 {rv} to {ty}")
                    self.lines.append(f"  store {ty} {tv}, ptr {p}, align {_align(ty)}")
            else:
                r = "%" + self.ctx.next_temp("rv")
                self.lines.append(f"  {r} = call i32 @getchar()")
                if ty == "i32":
                    self.lines.append(f"  store i32 {r}, ptr {p}, align 4")
                else:
                    t = "%" + self.ctx.next_temp("tr")
                    self.lines.append(f"  {t} = trunc i32 {r} to {ty}")
                    self.lines.append(f"  store {ty} {t}, ptr {p}, align {_align(ty)}")

    def _emit_snprintf_write(self, val: str, ty: str, fmt: str):
        buf = "%" + self.ctx.next_temp("sbuf")
        self.lines.append(f"  {buf} = alloca [64 x i8], align 1")
        p = "%" + self.ctx.next_temp("sp")
        self.lines.append(f"  {p} = getelementptr [64 x i8], ptr {buf}, i32 0, i32 0")
        self.lines.append(f"  call i32 (ptr, i64, ptr, ...) @snprintf(ptr {p}, i64 64, ptr {fmt}, {ty} {val})")
        ln = "%" + self.ctx.next_temp("slen")
        self.lines.append(f"  {ln} = call i64 @strlen(ptr {p})")
        self.lines.append(f"  call i64 @write(i32 1, ptr {p}, i64 {ln})")

    def _emit_ptr_arith_stmt(self, s: A.PtrArith):
        p, ty = self.ctx.locals[s.name]
        v = "%" + self.ctx.next_temp("ptr")
        self.lines.append(f"  {v} = load ptr, ptr {p}, align 8")
        nv = "%" + self.ctx.next_temp("nptr")
        self.lines.append(f"  {nv} = getelementptr i8, ptr {v}, i32 {s.delta}")
        self.lines.append(f"  store ptr {nv}, ptr {p}, align 8")

    def _emit_ptr_read(self, ptr_name: str, loc: Any):
        p, _ = self.ctx.locals[ptr_name]
        inn = self.ctx.ptr_inner.get(ptr_name, A.TypeRef("i8"))
        pb = "%" + self.ctx.next_temp("pb")
        self.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
        ity = _scalar_ty(inn)
        v = "%" + self.ctx.next_temp("v")
        self.lines.append(f"  {v} = load {ity}, ptr {pb}, align {_align(ity)}")
        cp, cty = self._get_current_cell_ptr()
        tv = "%" + self.ctx.next_temp("conv")
        if cty == ity: tv = v
        else: self.lines.append(f"  {tv} = sext {ity} {v} to {cty}")
        self.lines.append(f"  store {cty} {tv}, ptr {cp}, align {_align(cty)}")

    def _emit_ptr_write(self, s: A.PtrWrite):
        p, _ = self.ctx.locals[s.ptr]
        inn = self.ctx.ptr_inner.get(s.ptr, A.TypeRef("i8"))
        pb = "%" + self.ctx.next_temp("pb")
        self.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
        rv, rt = self._emit_expr(s.value, self.ctx)
        ity = _scalar_ty(inn)
        self.lines.append(f"  store {ity} {rv}, ptr {pb}, align {_align(ity)}")

    def _emit_alloc(self, s: A.AllocStmt):
        ity = _scalar_ty(s.ty)
        isz = _align(ity)
        total = s.count * isz
        r = "%" + self.ctx.next_temp("mem")
        self.lines.append(f"  {r} = call ptr @malloc(i64 {total})")
        if s.name:
            p = "%" + self.ctx.next_temp(f"v.{s.name}")
            self.lines.append(f"  {p} = alloca ptr, align 8")
            self.lines.append(f"  store ptr {r}, ptr {p}, align 8")
            self.ctx.locals[s.name] = (p, "ptr")
            self.ctx.ptr_inner[s.name] = s.ty

    def _emit_free(self, ptr_name: str):
        p, _ = self.ctx.locals[ptr_name]
        v = "%" + self.ctx.next_temp("v")
        self.lines.append(f"  {v} = load ptr, ptr {p}, align 8")
        self.lines.append(f"  call void @free(ptr {v})")

    def _emit_expr(self, e: A.Expr, ctx: LLVMContext) -> Tuple[str, str]:
        if isinstance(e, A.IntLit): return (str(e.value), "i32")
        if isinstance(e, A.FloatLit): return (str(e.value), "double")
        if isinstance(e, A.BoolLit): return ("1" if e.value else "0", "i1")
        if isinstance(e, A.Ident):
            if e.name in ctx.locals:
                p, ty = ctx.locals[e.name]
                t = "%" + ctx.next_temp(e.name)
                self.lines.append(f"  {t} = load {ty}, ptr {p}, align {_align(ty)}")
                return (t, ty)
            if e.name in self.seg_slots:
                return (self.seg_slots[e.name], "ptr")
            raise LLVMGenError(f"Unknown identifier {e.name}", e.loc)
        if isinstance(e, A.BinOp):
            l_v, l_t = self._emit_expr(e.left, ctx)
            r_v, r_t = self._emit_expr(e.right, ctx)
            res_t = "double" if l_t == "double" or r_t == "double" else "i32"
            if res_t == "double":
                if l_t != "double":
                    lt = "%" + ctx.next_temp("cvt")
                    self.lines.append(f"  {lt} = sitofp {l_t} {l_v} to double")
                    l_v = lt
                if r_t != "double":
                    rt = "%" + ctx.next_temp("cvt")
                    self.lines.append(f"  {rt} = sitofp {r_t} {r_v} to double")
                    r_v = rt
                op = {"+":"fadd","-":"fsub","*":"fmul","/":"fdiv","==":"fcmp oeq","!=":"fcmp une",">":"fcmp ogt",">=":"fcmp oge","<":"fcmp olt","<=":"fcmp ole"}[e.op]
                t = "%" + ctx.next_temp("bin")
                self.lines.append(f"  {t} = {op} double {l_v}, {r_v}")
                return (t, "i1" if e.op in ("==","!=",">",">=","<","<=") else "double")
            else:
                op = {"+":"add nsw","-":"sub nsw","*":"mul nsw","/":"sdiv","==":"icmp eq","!=":"icmp ne",">":"icmp sgt",">=":"icmp sge","<":"icmp slt","<=":"icmp sle"}[e.op]
                t = "%" + ctx.next_temp("bin")
                self.lines.append(f"  {t} = {op} i32 {l_v}, {r_v}")
                return (t, "i1" if e.op in ("==","!=",">",">=","<","<=") else "i32")
        if isinstance(e, A.Unary):
            if e.op == "-":
                v, ty = self._emit_expr(e.expr, ctx)
                t = "%" + ctx.next_temp("neg")
                if ty == "double": self.lines.append(f"  {t} = fneg double {v}")
                else: self.lines.append(f"  {t} = sub nsw {ty} 0, {v}")
                return (t, ty)
            if e.op == "*":
                 if not isinstance(e.expr, A.Ident): raise LLVMGenError("* only supports identifiers in SPEC", e.loc)
                 p, _ = ctx.locals[e.expr.name]
                 inn = ctx.ptr_inner.get(e.expr.name, A.TypeRef("i8"))
                 pb = "%" + ctx.next_temp("pb")
                 self.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
                 ity = _scalar_ty(inn)
                 t = "%" + ctx.next_temp("dr")
                 self.lines.append(f"  {t} = load {ity}, ptr {pb}, align {_align(ity)}")
                 return (t, ity)
            if e.op == "&":
                if not isinstance(e.expr, A.RefExpr): raise LLVMGenError("& expects a reference", e.loc)
                ptr, _ = self._gep(e.expr, ctx)
                return (ptr, "ptr")
        if isinstance(e, A.Call): return self._emit_call_expr(e, ctx)
        if isinstance(e, A.RefExpr):
            ptr, ty = self._gep(e, ctx)
            if e.parts and e.parts[0] == "@":
                 return (ptr, "ptr")
            t = "%" + ctx.next_temp("ref")
            self.lines.append(f"  {t} = load {ty}, ptr {ptr}, align {_align(ty)}")
            return (t, ty)
        raise LLVMGenError(f"unsupported expr {type(e)}")

    def _emit_call_expr(self, c: A.Call, ctx: LLVMContext) -> Tuple[str, str]:
        if c.name == "sqrt":
            v, ty = self._emit_expr(c.args[0], ctx)
            t = "%" + ctx.next_temp("sqrt")
            self.lines.append(f"  {t} = call double @llvm.sqrt.f64(double {v})")
            return (t, "double")
        if getattr(self.mod, "use_linux_stdlib", False) and c.name in _LINUX_LIBC:
            ret_t, arg_ts = _LINUX_LIBC[c.name]
            parts = []
            for i, arg in enumerate(c.args):
                av, at = self._emit_expr(arg, ctx)
                if i < len(arg_ts):
                    target_t = arg_ts[i]
                    if target_t == "i64" and at == "i32":
                         cv = "%" + ctx.next_temp("zext")
                         self.lines.append(f"  {cv} = zext i32 {av} to i64")
                         av = cv; at = "i64"
                parts.append(f"{at} {av}")
            t = "%" + ctx.next_temp("call")
            if ret_t == "void":
                self.lines.append(f"  call void @{c.name}({', '.join(parts)})")
                return ("", "void")
            self.lines.append(f"  {t} = call {ret_t} @{c.name}({', '.join(parts)})")
            return (t, ret_t)
        fd = self.fns.get(c.name)
        if not fd: raise LLVMGenError(f"Unknown function {c.name}", c.loc)
        parts = []
        for i, (_, pt) in enumerate(fd.params):
            av, at = self._emit_expr(c.args[i], ctx)
            target_t = _scalar_ty(pt)
            if target_t == "double" and at == "i32":
                cv = "%" + ctx.next_temp("conv")
                self.lines.append(f"  {cv} = sitofp i32 {av} to double")
                av = cv; at = "double"
            parts.append(f"{at} {av}")
        rt = _scalar_ty(fd.ret)
        if rt == "void":
            self.lines.append(f"  call void @{c.name}({', '.join(parts)})")
            return ("", "void")
        t = "%" + ctx.next_temp("call")
        self.lines.append(f"  {t} = call {rt} @{c.name}({', '.join(parts)})")
        return (t, rt)

    def _gep(self, r: A.RefExpr, ctx: LLVMContext) -> Tuple[str, str]:
        if r.parts and r.parts[0] == "@":
            base_n = r.parts[1]
            p = self.seg_slots.get(base_n)
            seg = self.global_segs.get(base_n)
            lty = _scalar_ty(seg.elem_type)
            return (p, lty)
        head = str(r.parts[0])
        if head in ctx.locals:
            p, ty = ctx.locals[head]
            lty = ty
            slot = 1
            while slot < len(r.parts):
                 sub = r.parts[slot]
                 if isinstance(sub, A.Expr):
                     idx_v, _ = self._emit_expr(sub, ctx)
                     if lty == "ptr":
                         pb = "%" + ctx.next_temp("pb")
                         self.lines.append(f"  {pb} = load ptr, ptr {p}, align 8")
                         p = "%" + ctx.next_temp("ptr")
                         inn = ctx.ptr_inner.get(head, A.TypeRef("i8"))
                         lty = _scalar_ty(inn)
                         self.lines.append(f"  {p} = getelementptr {lty}, ptr {pb}, i32 {idx_v}")
                     elif lty.startswith("["):
                         inn_ty = lty[lty.find("x")+1 : -1].strip()
                         p_old = p
                         p = "%" + ctx.next_temp("ptr")
                         self.lines.append(f"  {p} = getelementptr {lty}, ptr {p_old}, i32 0, i32 {idx_v}")
                         lty = inn_ty
                 elif isinstance(sub, str):
                     # Field access
                     s_name = ""
                     if lty.startswith("%struct."):
                         s_name = lty[8:]
                     elif lty == "ptr":
                         inn = self.ctx.ptr_inner.get(head, A.TypeRef("i8"))
                         s_name = inn.name
                     
                     if not s_name or s_name not in self.structs:
                         raise LLVMGenError(f"Cannot resolve struct field {sub} on type {lty}")
                     
                     s_decl = self.structs[s_name]
                     field_idx = -1
                     for i, (fn, ft) in enumerate(s_decl.fields):
                         if fn == sub:
                             field_idx = i
                             lty = _scalar_ty(ft)
                             # Update ptr_inner if the new type is a pointer
                             if ft.name == "ptr":
                                 self.ctx.ptr_inner[f"__tmp_{s_name}_{sub}"] = ft.inner or A.TypeRef("i8")
                                 head = f"__tmp_{s_name}_{sub}"
                             break
                     if field_idx == -1:
                          raise LLVMGenError(f"Field {sub} not found in struct {s_name}")
                     p_old = p
                     p = f"%{self.ctx.next_temp('ptr')}"
                     self.lines.append(f"  {p} = getelementptr %struct.{s_name}, ptr {p_old}, i32 0, i32 {field_idx}")
                 slot += 1
            return (p, lty)
        if head in self.seg_slots:
            p = self.seg_slots[head]
            seg = self.global_segs[head]
            lty = _scalar_ty(seg.elem_type)
            p_res = p
            full_ty = f"[{seg.length} x {lty}]"
            slot = 1
            while slot < len(r.parts):
                 sub = r.parts[slot]
                 if isinstance(sub, A.Expr):
                     idx_v, _ = self._emit_expr(sub, ctx)
                     p_old = p_res
                     p_res = "%" + ctx.next_temp("ptr")
                     self.lines.append(f"  {p_res} = getelementptr {full_ty}, ptr {p_old}, i32 0, i32 {idx_v}")
                     full_ty = lty
                 slot += 1
            return (p_res, lty)
        raise LLVMGenError(f"Cannot resolve reference {r.parts}")

    def _get_current_cell_ptr(self) -> Tuple[str, str]:
        bp = "%" + self.ctx.next_temp("cseg")
        self.lines.append(f"  {bp} = load ptr, ptr @__cseg, align 8")
        idx = "%" + self.ctx.next_temp("cslot")
        self.lines.append(f"  {idx} = load i32, ptr @__cslot, align 4")
        lty = self.ctx.cursor_type
        p = "%" + self.ctx.next_temp("cptr")
        self.lines.append(f"  {p} = getelementptr {lty}, ptr {bp}, i32 {idx}")
        return (p, lty)

    def _emit_cond(self, s: A.Cond) -> str:
        p, ty = self._get_current_cell_ptr()
        v = "%" + self.ctx.next_temp("cv")
        self.lines.append(f"  {v} = load {ty}, ptr {p}, align {_align(ty)}")
        t = "%" + self.ctx.next_temp("cond")
        if s.kind == ">0": self.lines.append(f"  {t} = icmp sgt {ty} {v}, 0")
        elif s.kind == "<0": self.lines.append(f"  {t} = icmp slt {ty} {v}, 0")
        elif s.kind == "==0": self.lines.append(f"  {t} = icmp eq {ty} {v}, 0")
        elif s.kind == "!=0": self.lines.append(f"  {t} = icmp ne {ty} {v}, 0")
        elif s.kind == ">N": self.lines.append(f"  {t} = icmp sgt {ty} {v}, {s.imm}")
        elif s.kind == "<N": self.lines.append(f"  {t} = icmp slt {ty} {v}, {s.imm}")
        elif s.kind == "==N": self.lines.append(f"  {t} = icmp eq {ty} {v}, {s.imm}")
        elif s.kind == "!=N": self.lines.append(f"  {t} = icmp ne {ty} {v}, {s.imm}")
        return t

def emit_llvm_ir(mod: A.Module, target: str | None = None) -> str:
    if target is None:
        target = f"{platform.machine()}-pc-linux-gnu"
    return LLVMEmitterVisitor(mod, target).emit()
