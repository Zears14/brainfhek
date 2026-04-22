from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from bf2.core import ast as A
from bf2.core.errors import BF2TypeError, SourceLoc
from bf2.core.visitor import ASTVisitor
from bf2.core.memory import build_struct_layout
from bf2.compiler.diagnostics import DiagnosticCollector
from bf2.core.analysis import find_segment_deps

class TypeChecker(ASTVisitor):
    """Semantic analysis and type checking for Brainfhek."""

    def __init__(self, mod: A.Module, diag: Optional[DiagnosticCollector] = None):
        self.mod = mod
        self.diag = diag or DiagnosticCollector()
        self.structs: Dict[str, A.StructDecl] = {}
        self.segs: Dict[str, Tuple[A.TypeRef, int]] = {}
        self.fns: Dict[str, Tuple[List[Tuple[str, A.TypeRef]], A.TypeRef]] = {}
        self.scopes: List[Dict[str, A.TypeRef]] = []
        self._used_vars: Set[str] = set()
        self.current_fn: Optional[A.FunctionDef] = None
        self.use_linux_stdlib = getattr(mod, "use_linux_stdlib", False)
        self.mod_links: Dict[str, Set[str]] = {}

    def check(self) -> None:
        # Pre-collection
        for item in self.mod.items:
            if isinstance(item, A.StructDecl):
                self.structs[item.name] = item
            if isinstance(item, A.SegmentDecl):
                if item.name in self.segs:
                     raise BF2TypeError(f"redefinition of segment '{item.name}'", item.loc)
                self.segs[item.name] = (item.elem_type, item.length)
            elif isinstance(item, A.FunctionDef):
                if item.name in self.fns:
                     raise BF2TypeError(f"redefinition of function '{item.name}'", item.loc)
                self.fns[item.name] = ([(p[0], p[1]) for p in item.params], item.ret)
            if isinstance(item, A.SegmentDecl) and item.init:
                self.mod_links[item.name] = find_segment_deps(item.init)
        
        # Check items
        for item in self.mod.items:
            self.visit(item)

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _declare(self, name: str, ty: A.TypeRef, loc: SourceLoc) -> None:
        if name in self.scopes[-1]:
            raise BF2TypeError(f"redefinition of '{name}'", loc)
        # Check for shadowing in outer scopes
        for scope in self.scopes[:-1]:
            if name in scope:
                self.diag.warn("shadow", f"variable '{name}' shadows outer declaration", loc)
                break
        self.scopes[-1][name] = ty

    def _resolve(self, name: str, loc: SourceLoc) -> A.TypeRef:
        for scope in reversed(self.scopes):
            if name in scope:
                self._used_vars.add(name)
                return scope[name]
        if name in self.segs:
            return A.TypeRef("ptr", self.segs[name][0])
        raise BF2TypeError(f"undefined identifier '{name}'", loc)

    # --- Visitor Implementation ---

    def visit_struct_decl(self, node: A.StructDecl) -> None:
        pass

    def visit_segment_decl(self, node: A.SegmentDecl) -> None:
        self.segs[node.name] = (node.elem_type, node.length)
        if node.init:
            deps = self.mod_links.get(node.name, set())
            for d in deps:
                if d == node.name:
                    raise BF2TypeError(f"segment '{node.name}' cannot depend on itself", node.loc)
                if d not in self.segs:
                    # Could be a local if we support it, but segments are usually global
                    self.diag.warn("sem", f"linked segment '{node.name}' depends on unknown identifier '{d}'", node.loc)
                else:
                    d_ty, d_len = self.segs[d]
                    if node.length > d_len:
                        self.diag.warn("ub", f"linked segment '{node.name}' (len {node.length}) is larger than its dependency '{d}' (len {d_len}) — potential out-of-bounds access during reactive updates", node.loc)
            
            # Check for circularity
            self._check_circular_links(node.name, set(), node.loc)

    def _check_circular_links(self, name: str, path: Set[str], loc: SourceLoc) -> None:
        if name in path:
            raise BF2TypeError(f"circular dependency detected involving segment '{name}'", loc)
        
        if name not in self.mod_links:
            return
            
        path.add(name)
        for dep in self.mod_links[name]:
            self._check_circular_links(dep, path, loc)
        path.remove(name)

    def visit_segment_stmt(self, node: A.SegmentStmt) -> None:
        self.visit(node.decl)

    def visit_struct_stmt(self, node: A.StructStmt) -> None:
        self.structs[node.decl.name] = node.decl

    def visit_function_def(self, node: A.FunctionDef) -> None:
        self.current_fn = node
        self._push_scope()
        for name, ty in node.params:
            self._declare(name, ty, node.loc)
        self.visit(node.body)
        self._pop_scope()
        self.current_fn = None

    def visit_block(self, node: A.Block) -> None:
        self._push_scope()
        saw_ret = False
        for stmt in node.stmts:
            if saw_ret:
                self.diag.warn(
                    "unreachable",
                    "unreachable code after return statement",
                    getattr(stmt, "loc", None),
                )
                break
            self.visit(stmt)
            if isinstance(stmt, A.RetStmt):
                saw_ret = True
        # Check for unused variables in this scope
        for name in self.scopes[-1]:
            if name not in self._used_vars:
                self.diag.warn("unused", f"variable '{name}' is declared but never used")
        self._pop_scope()

    def visit_var_decl(self, node: A.VarDecl) -> None:
        if node.init:
            self.visit(node.init)
        self._declare(node.name, node.ty, node.loc)

    def visit_ptr_decl(self, node: A.PtrDecl) -> None:
        if node.init:
            self.visit(node.init)
        self._declare(node.name, A.TypeRef("ptr", node.inner), node.loc)

    def visit_ret_stmt(self, node: A.RetStmt) -> None:
        if node.value:
            self.visit(node.value)

    def visit_assign_stmt(self, node: A.AssignStmt) -> None:
        lt = self.visit(node.lhs)
        rt = self.visit(node.rhs)
        if lt.name != rt.name and not (lt.name in ("i32", "i8", "i64", "f32", "f64") and rt.name in ("i32", "i8", "i64", "f32", "f64")):
             self.diag.warn("types", f"assignment between potentially incompatible types '{rt.name}' and '{lt.name}'", node.loc)

    def visit_if_stmt(self, node: A.IfStmt) -> None:
        self.visit(node.then)
        if node.els:
            self.visit(node.els)

    def visit_loop_bf(self, node: A.LoopBF) -> None:
        # Detect potential infinite loops: no cell mutation or cursor move
        has_mutation = False
        for stmt in node.body.stmts:
            if isinstance(stmt, (A.CellArith, A.CellArithRef, A.CellAssignLit,
                                 A.MoveRel, A.RetStmt, A.AssignStmt)):
                has_mutation = True
                break
        if not has_mutation:
            self.diag.warn(
                "ub",
                "loop body has no cell mutation or cursor move — possible infinite loop",
                node.loc,
            )
        self.visit(node.body)

    def visit_loop_counted(self, node: A.LoopCounted) -> None:
        self.visit(node.body)

    def visit_ptr_read(self, node: A.PtrRead) -> None:
        t = self._resolve(node.name, node.loc)
        if t.name != "ptr":
             self.diag.warn("types", f"'ptrread' on non-pointer type '{t.name}'", node.loc)

    def visit_ptr_write(self, node: A.PtrWrite) -> None:
        t = self._resolve(node.name, node.loc)
        if t.name != "ptr":
             self.diag.warn("types", f"pointer write on non-pointer type '{t.name}'", node.loc)
        self.visit(node.expr)

    def visit_free_stmt(self, node: A.FreeStmt) -> None:
        t = self._resolve(node.name, node.loc)
        if t.name != "ptr":
             self.diag.warn("types", f"'free' on non-pointer type '{t.name}'", node.loc)

    def visit_alloc_stmt(self, node: A.AllocStmt) -> None:
        if node.size > 1024 * 1024 * 1024:
             self.diag.warn("unsafe", f"large allocation of {node.size} bytes", node.loc)
        if node.name:
             self._declare(node.name, A.TypeRef("ptr", node.ty), node.loc)

    def visit_call_stmt(self, node: A.CallStmt) -> None:
        self.visit(node.call)

    def visit_io_stmt(self, node: A.IOStmt) -> None:
        if node.expr:
            self.visit(node.expr)

    def visit_alloc_stmt(self, node: A.AllocStmt) -> None:
        if node.name:
            self._declare(node.name, A.TypeRef("ptr", node.ty), node.loc)

    def visit_int_lit(self, node: A.IntLit) -> A.TypeRef:
        return A.TypeRef("i32")

    def visit_float_lit(self, node: A.FloatLit) -> A.TypeRef:
        return A.TypeRef("f64")

    def visit_bool_lit(self, node: A.BoolLit) -> A.TypeRef:
        return A.TypeRef("bool")

    def visit_string_lit(self, node: A.StringLit) -> A.TypeRef:
        return A.TypeRef("ptr", A.TypeRef("i8"))

    def visit_unary(self, node: A.Unary) -> A.TypeRef:
        t = self.visit(node.expr)
        if node.op == "*":
             if t.name != "ptr":
                 raise BF2TypeError(f"cannot dereference non-pointer type '{t.name}'", node.loc)
             return t.ty
        if node.op == "&":
             return A.TypeRef("ptr", t)
        return t

    def visit_ident(self, node: A.Ident) -> A.TypeRef:
        return self._resolve(node.name, node.loc)

    def visit_ref_expr(self, node: A.RefExpr) -> A.TypeRef:
        parts = node.parts
        if parts and parts[0] == "@":
            return A.TypeRef("i8")
        if not parts:
            raise BF2TypeError("empty reference", node.loc)
        
        head = parts[0]
        if not isinstance(head, str):
            raise BF2TypeError("reference must start with name", node.loc)
        
        t = self._resolve(head, node.loc)
        seg_length = self.segs[head][1] if head in self.segs else None
        
        slot = 1
        while slot < len(parts):
            p = parts[slot]
            if isinstance(p, A.IntLit):
                if seg_length is not None and slot == 1:
                    if p.value < 0 or p.value >= seg_length:
                        self.diag.warn("bounds", f"index {p.value} is out of bounds for segment '{head}' of length {seg_length}", node.loc)
                # Ensure the index expression itself is typechecked
                self.visit(p)
            elif isinstance(p, A.Expr):
                self.visit(p)
            elif isinstance(p, str):
                if t.name == "ptr" and t.inner:
                    inn = t.inner
                    if inn.name in self.structs:
                        sl = build_struct_layout(self.structs[inn.name])
                        for fn, ft in sl.fields:
                            if fn == p:
                                t = ft
                                break
                elif t.name in self.structs:
                    sl = build_struct_layout(self.structs[t.name])
                    for fn, ft in sl.fields:
                        if fn == p:
                            t = ft
                            break
            slot += 1
        return t

    def _is_numeric(self, ty: A.TypeRef) -> bool:
        return ty.name in ("i8", "i32", "i64", "f32", "f64")

    def _is_float(self, ty: A.TypeRef) -> bool:
        return ty.name in ("f32", "f64")

    def _get_const_int(self, node: A.Expr) -> int | None:
        if isinstance(node, A.IntLit):
            return node.value
        if isinstance(node, A.Unary) and node.op == "-":
            val = self._get_const_int(node.expr)
            return -val if val is not None else None
        return None

    def visit_bin_op(self, node: A.BinOp) -> A.TypeRef:
        lt = self.visit(node.left)
        rt = self.visit(node.right)
        
        # Detect division by zero
        if node.op == "/" and isinstance(node.right, A.IntLit) and node.right.value == 0:
            self.diag.warn("ub", "division by zero", node.loc)
        
        # Detect bitwise on floats
        if node.op in ("&", "|", "^", "<<", ">>"):
            if self._is_float(lt) or self._is_float(rt):
                self.diag.warn("ub", f"bitwise operation '{node.op}' on floating-point types is undefined", node.loc)
            
            # Shift checks
            if node.op in ("<<", ">>"):
                rv = self._get_const_int(node.right)
                if rv is not None:
                    if rv < 0:
                        self.diag.warn("ub", f"negative shift count {rv}", node.loc)
                    elif rv >= 64:
                        self.diag.warn("ub", f"shift count {rv} is too large", node.loc)

        if self._is_float(lt) or self._is_float(rt):
             return A.TypeRef("f64")
        return A.TypeRef("i32")

    def visit_ret_stmt(self, node: A.RetStmt) -> None:
        if node.value:
            self.visit(node.value)

    def visit_expr_stmt(self, node: A.ExprStmt) -> None:
        if node.expr:
            self.visit(node.expr)

    def visit_call_stmt(self, node: A.CallStmt) -> None:
        self.visit(node.call)

    def visit_call(self, node: A.Call) -> A.TypeRef:
        arg_tys = [self.visit(a) for a in node.args]
        
        if node.name not in self.fns:
            if self.use_linux_stdlib and node.name in ("write", "read", "open", "close", "exit", "fork", "getpid", "strlen", "nanosleep"):
                return A.TypeRef("i32")
            if node.name in ("sqrt", "putchar", "getchar", "scanf", "printf"):
                return A.TypeRef("i32")
            raise BF2TypeError(f"unknown function: {node.name}", node.loc)
        
        params, ret = self.fns[node.name]
        if len(arg_tys) != len(params):
             # Some functions might be vararg but BF2 doesn't explicitly track it yet
             # Only warn for now if it's not a known vararg-ish function
             if node.name not in ("printf", "scanf"):
                 self.diag.warn("types", f"function '{node.name}' expects {len(params)} arguments but got {len(arg_tys)}", node.loc)
        
        return ret

def check_module(
    mod: A.Module,
    diag: Optional[DiagnosticCollector] = None,
) -> DiagnosticCollector:
    """Run semantic analysis and return the diagnostic collector."""
    d = diag or DiagnosticCollector()
    TypeChecker(mod, d).check()
    return d
