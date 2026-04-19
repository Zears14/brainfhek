from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from bf2.core import ast as A

class LLVMGenError(Exception):
    def __init__(self, msg: str, loc: Optional[A.SourceLoc] = None):
        super().__init__(msg)
        self.loc = loc

class LLVMContext:
    """Emission state for a single function/reactor in the LLVM backend."""
    __slots__ = (
        "ret_ty",
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
    )

    def __init__(self, ret_ty: str) -> None:
        self.ret_ty = ret_ty
        self.locals: Dict[str, Tuple[str, str]] = {}
        self.cursor_slot_ptr = ""
        self.cursor_seg = "__bf"
        self.ptr_struct: Dict[str, str] = {}
        self.local_segs: Dict[str, A.SegmentDecl] = {}
        self.local_seg_alloca: Dict[str, str] = {}
        self.local_structs: Dict[str, A.StructDecl] = {}
        self.label_map: Dict[str, str] = {}
        self.pending_labels: List[str] = []
        self.ptr_inner: Dict[str, A.TypeRef] = {}
        self.static_cslot: Optional[int] = 0
        self.next_temp_id = 0
        self.next_label_id = 0
        self.cursor_type = "i8"

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
