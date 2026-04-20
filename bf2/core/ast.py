from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING

from bf2.core.errors import SourceLoc

if TYPE_CHECKING:
    from bf2.core.visitor import ASTVisitor


@dataclass
class TypeRef:
    name: str
    inner: Optional[TypeRef] = None

    def __str__(self) -> str:
        if self.name == "ptr" and self.inner:
            return f"ptr<{self.inner}>"
        return self.name


@dataclass
class Module:
    items: list[Any]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))
    #: Set when sources use ``#include`` of the Linux libc stdlib header.
    use_linux_stdlib: bool = False

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_module(self)


@dataclass
class StructDecl:
    name: str
    fields: list[tuple[str, TypeRef]]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_struct_decl(self)


@dataclass
class SegmentDecl:
    name: str
    elem_type: TypeRef
    length: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_segment_decl(self)


@dataclass
class FunctionDef:
    name: str
    params: list[tuple[str, TypeRef]]
    ret: TypeRef
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_def(self)


@dataclass
class ReactorDef:
    target: RefExpr
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_reactor_def(self)


@dataclass
class Block:
    stmts: list[Any]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_block(self)


# --- expressions ---


@dataclass
class RefExpr:
    """seg[i], seg.field chain, @name"""
    parts: list[Union[str, int, "Expr"]]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ref_expr(self)


@dataclass
class Ident:
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ident(self)


@dataclass
class IntLit:
    value: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_int_lit(self)


@dataclass
class FloatLit:
    value: float
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_float_lit(self)


@dataclass
class BoolLit:
    value: bool
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_bool_lit(self)


@dataclass
class StringLit:
    value: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_string_lit(self)


@dataclass
class Unary:
    op: str
    expr: "Expr"
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary(self)


@dataclass
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_bin_op(self)


@dataclass
class Call:
    name: str
    args: list["Expr"]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_call(self)


Expr = Union[RefExpr, Ident, IntLit, FloatLit, BoolLit, StringLit, Unary, BinOp, Call]


# --- statements ---


@dataclass
class MoveOp:
    target: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_move_op(self)


@dataclass
class MoveRel:
    delta: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_move_rel(self)


@dataclass
class CellArith:
    op: str
    amount: Optional[Union[int, float]]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_cell_arith(self)


@dataclass
class CellArithRef:
    target: RefExpr
    op: str
    amount: Union[int, float]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_cell_arith_ref(self)


@dataclass
class CellAssignLit:
    value: Union[int, float, bool]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_cell_assign_lit(self)


@dataclass
class LoadOp:
    src: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_load_op(self)


@dataclass
class StoreOp:
    dst: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_store_op(self)


@dataclass
class SwapOp:
    other: RefExpr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_swap_op(self)


@dataclass
class AssignStmt:
    lhs: RefExpr
    rhs: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_assign_stmt(self)


@dataclass
class VarDecl:
    ty: TypeRef
    name: str
    init: Optional[Expr]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_var_decl(self)


@dataclass
class PtrDecl:
    inner: TypeRef
    name: str
    init: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ptr_decl(self)


@dataclass
class PtrArith:
    name: str
    delta: int
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ptr_arith(self)


@dataclass
class PtrWrite:
    ptr: str
    value: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ptr_write(self)


@dataclass
class PtrRead:
    ptr: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ptr_read(self)


@dataclass
class Cond:
    kind: str
    imm: Optional[Union[int, float]] = None
    expr: Optional["Expr"] = None


@dataclass
class IfStmt:
    cond: Cond
    then: Block
    els: Optional[Block]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_if_stmt(self)


@dataclass
class LoopBF:
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_loop_bf(self)


@dataclass
class LoopCounted:
    count: int
    body: Block
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_loop_counted(self)


@dataclass
class LabelStmt:
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_label_stmt(self)


@dataclass
class JumpStmt:
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_jump_stmt(self)


@dataclass
class IOStmt:
    kind: str
    expr: Optional[Expr] = None
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_io_stmt(self)


@dataclass
class CallStmt:
    call: Call
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_call_stmt(self)


@dataclass
class RetStmt:
    value: Optional[Expr]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_ret_stmt(self)


@dataclass
class AllocStmt:
    ty: TypeRef
    count: int
    name: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_alloc_stmt(self)


@dataclass
class FreeStmt:
    ptr: str
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_free_stmt(self)


@dataclass
class ExprStmt:
    expr: Expr
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_expr_stmt(self)


@dataclass
class SegmentStmt:
    decl: SegmentDecl
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_segment_stmt(self)


@dataclass
class StructStmt:
    decl: StructDecl
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_struct_stmt(self)
