from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from bf2.errors import SourceLoc


@dataclass
class TypeRef:
    name: str
    inner: Optional[TypeRef] = None

    def __str__(self) -> str:
        if self.name == "ptr" and self.inner:
            return f"ptr<{self.inner}>"
        return self.name

    def to_llvm_ir_hint(self) -> str:
        if self.name == "ptr" and self.inner:
            return f"{self.inner.to_llvm_ir_hint()}*"
        m = {"i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64", "f32": "float", "f64": "double", "bool": "i1"}
        return m.get(self.name, self.name)


@dataclass
class Module:
    items: list[Any]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))
    #: Set when sources use ``#include`` of the Linux libc stdlib header.
    use_linux_stdlib: bool = False

    def to_llvm_ir_hint(self) -> str:
        return "; module"


@dataclass
class StructDecl:
    name: str
    fields: list[tuple[str, TypeRef]]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"%{self.name} = type {{ ... }}"


@dataclass
class SegmentDecl:
    name: str
    elem_type: TypeRef
    length: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"%{self.name} = alloca [{self.length} x {self.elem_type.to_llvm_ir_hint()}]"


@dataclass
class FunctionDef:
    name: str
    params: list[tuple[str, TypeRef]]
    ret: TypeRef
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        args = ", ".join(f"{t.to_llvm_ir_hint()} %{n}" for n, t in self.params)
        return f"define {self.ret.to_llvm_ir_hint()} @{self.name}({args})"


@dataclass
class ReactorDef:
    target: RefExpr
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "store ... + call void @reactor_..."


@dataclass
class Block:
    stmts: list[Any]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "{ ... basic blocks ... }"


# --- expressions ---


@dataclass
class RefExpr:
    """seg[i], seg.field chain, @name"""

    parts: list[Union[str, int, "Expr"]]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "getelementptr + load/store"


@dataclass
class Ident:
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "%" + self.name


@dataclass
class IntLit:
    value: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return str(self.value)


@dataclass
class FloatLit:
    value: float
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return str(self.value)


@dataclass
class BoolLit:
    value: bool
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "i1 true" if self.value else "i1 false"


@dataclass
class Unary:
    op: str
    expr: "Expr"
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return {"-": "sub 0, %v", "*": "load", "&": "getelementptr"}.get(self.op, self.op)


@dataclass
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        m = {"+": "add nsw", "-": "sub nsw", "*": "mul nsw", "/": "sdiv", "%": "srem"}
        return m.get(self.op, "fcmp/fadd") + " ..."


@dataclass
class Call:
    name: str
    args: list["Expr"]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"call @{self.name}(...)"


Expr = Union[RefExpr, Ident, IntLit, FloatLit, BoolLit, Unary, BinOp, Call]


# --- statements ---


@dataclass
class MoveOp:
    target: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "ptr = getelementptr ..."


@dataclass
class MoveRel:
    delta: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "ptr += delta * sizeof(cell)"


@dataclass
class CellArith:
    op: str
    amount: Optional[Union[int, float]]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return {"+": "add nsw", "-": "sub nsw", "*": "mul nsw", "/": "sdiv"}.get(self.op, self.op) + " cell, imm"


@dataclass
class CellArithRef:
    target: RefExpr
    op: str
    amount: Union[int, float]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "load addr; op; store"


@dataclass
class CellAssignLit:
    value: Union[int, float, bool]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "store imm, ptr %cell"


@dataclass
class LoadOp:
    src: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "%r = load T, T* addr; store %r, cellptr"


@dataclass
class StoreOp:
    dst: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "store cellval, T* addr"


@dataclass
class SwapOp:
    other: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "load both; store swapped"


@dataclass
class AssignStmt:
    lhs: RefExpr
    rhs: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "store rhs, addr"


@dataclass
class VarDecl:
    ty: TypeRef
    name: str
    init: Optional[Expr]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"%{self.name} = alloca {self.ty.to_llvm_ir_hint()}"


@dataclass
class PtrDecl:
    inner: TypeRef
    name: str
    init: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"%{self.name} = getelementptr ..."


@dataclass
class PtrArith:
    name: str
    delta: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "getelementptr inbounds T, T* %p, i64 delta"


@dataclass
class PtrWrite:
    ptr: str
    value: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "store T %v, T* %p"


@dataclass
class PtrRead:
    ptr: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "%r = load T, T* %p"


@dataclass
class Cond:
    kind: str
    imm: Optional[Union[int, float]] = None

    def to_llvm_ir_hint(self) -> str:
        return "icmp ..."


@dataclass
class IfStmt:
    cond: Cond
    then: Block
    els: Optional[Block]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "br i1 %c, label %then, label %else"


@dataclass
class LoopBF:
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "loop: phi; br i1 %nonzero, body, exit"


@dataclass
class LoopCounted:
    count: int
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "repeat N: br loop"


@dataclass
class LabelStmt:
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"{self.name}:"


@dataclass
class JumpStmt:
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return f"br label %{self.name}"


@dataclass
class IOStmt:
    kind: str
    expr: Optional[Expr] = None
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        m = {
            ".": "call putchar",
            ",": "call getchar",
            ".i": "call bf2_print_int",
            ".f": "call bf2_print_float",
            ".s": "call bf2_print_str",
            ",s": "call bf2_read_line",
        }
        return m.get(self.kind, "io")


@dataclass
class CallStmt:
    call: Call
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return self.call.to_llvm_ir_hint()


@dataclass
class RetStmt:
    value: Optional[Expr]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "ret T %v"


@dataclass
class AllocStmt:
    ty: TypeRef
    count: int
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "%p = call i8* @malloc(i64 n); bitcast"


@dataclass
class FreeStmt:
    ptr: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "call void @free(i8* %p)"


@dataclass
class ExprStmt:
    expr: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return "expr as stmt"


@dataclass
class SegmentStmt:
    decl: SegmentDecl
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return self.decl.to_llvm_ir_hint()


@dataclass
class StructStmt:
    decl: StructDecl
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def to_llvm_ir_hint(self) -> str:
        return self.decl.to_llvm_ir_hint()
