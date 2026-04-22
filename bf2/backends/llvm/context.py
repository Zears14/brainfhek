from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

from bf2.core import ast as A
from bf2.backends.llvm.types import to_ir_type, Int8

if TYPE_CHECKING:
    from llvmlite import ir


class LLVMGenError(Exception):
    def __init__(self, msg: str, loc: Optional[A.SourceLoc] = None):
        super().__init__(msg)
        self.loc = loc


class LLVMContext:
    """Emission state for a single function/reactor in the LLVM backend."""

    __slots__ = (
        "ret_ty",
        "builder",
        "locals",
        "cursor_slot_ptr",
        "cursor_seg",
        "ptr_struct",
        "local_segs",
        "local_seg_alloca",
        "local_structs",
        "label_map",
        "pending_labels",
        "ptr_inner",
        "static_cslot",
        "next_temp_id",
        "next_label_id",
        "cursor_type",
        "blocks",
        "_alloca_cursor",
        "cursor_seg_name",
        "cursor_index_v",
    )

    def __init__(self, ret_ty: A.TypeRef, builder: ir.IRBuilder | None = None) -> None:
        self.ret_ty = to_ir_type(ret_ty) if isinstance(ret_ty, A.TypeRef) else ret_ty
        self.builder = builder
        self.locals: Dict[str, tuple[ir.Value, ir.Type]] = {}
        self.cursor_slot_ptr = ""
        self.cursor_seg = "__bf"
        self.ptr_struct: Dict[str, str] = {}
        self.local_segs: Dict[str, A.SegmentDecl] = {}
        self.local_seg_alloca: Dict[str, str] = {}
        self.local_structs: Dict[str, A.StructDecl] = {}
        self.label_map: Dict[str, str] = {}
        self.pending_labels: list[str] = []
        self.ptr_inner: Dict[str, A.TypeRef] = {}
        self.static_cslot: Optional[int] = 0
        self.next_temp_id = 0
        self.next_label_id = 0
        self.cursor_type: ir.Type = Int8
        self.blocks: Dict[str, Any] = {}
        self._alloca_cursor = None
        self.cursor_seg_name: Optional[str] = None
        self.cursor_index_v: Optional[ir.Value] = None

    def next_temp(self, suffix: str = "") -> str:
        """Returns a name WITHOUT the leading %."""
        self.next_temp_id += 1
        s = f".{suffix}" if suffix else ""
        return f"t{self.next_temp_id}{s}"

    def next_label(self, hint: str) -> str:
        """Returns a name WITHOUT the leading %."""
        self.next_label_id += 1
        return f"{hint}.{self.next_label_id}"

    def invalidate_static_cslot(self) -> None:
        """Called when the cursor movement becomes dynamic or enters a fresh block."""
        self.static_cslot = None

    def set_static_cslot(self, slot: int) -> None:
        self.static_cslot = slot

    def hoist_alloca(self, ty: ir.Type, name: str = "") -> ir.Value:
        """Create an alloca in the entry block of the current function."""
        entry = self.builder.function.entry_basic_block
        with self.builder.goto_block(entry):
            if self._alloca_cursor is not None:
                self.builder.position_after(self._alloca_cursor)
            elif entry.instructions:
                self.builder.position_before(entry.instructions[0])
            else:
                self.builder.position_at_start(entry)
            
            inst = self.builder.alloca(ty, name=name)
            self._alloca_cursor = inst
            return inst
