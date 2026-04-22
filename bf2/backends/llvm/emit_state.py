"""Shared emission state threaded through all LLVM codegen sub-modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bf2.core import ast as A

if TYPE_CHECKING:
    pass


@dataclass
class EmitState:
    """Mutable state container for a single LLVM IR emission pass.

    Uses llvmlite ir.Module and ir.IRBuilder for type-safe IR generation.
    """

    mod: A.Module
    target: str
    module: Any = field(default=None)
    builder: Any = field(default=None)

    locals: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    seg_slots: dict[str, Any] = field(default_factory=dict)

    structs: dict[str, A.StructDecl] = field(default_factory=dict)
    fns: dict[str, A.FunctionDef] = field(default_factory=dict)
    global_segs: dict[str, A.SegmentDecl] = field(default_factory=dict)
    links: dict[str, list[A.SegmentDecl]] = field(default_factory=dict)
    one_time_links: list[A.SegmentDecl] = field(default_factory=list)
    link_active_gv: dict[str, Any] = field(default_factory=dict)
    watches: list[tuple[str, int, A.ReactorDef]] = field(default_factory=list)
    watch_blocks: dict[str, Any] = field(default_factory=dict)

    loop_metadata: list[str] = field(default_factory=list)
    next_metadata_id: int = 0

    fn_attrs: dict[str, str] = field(default_factory=dict)

    string_constants: dict[str, str] = field(default_factory=dict)
    next_str_id: int = 0

    declared_externs: set[str] = field(default_factory=set)

    def alloc_metadata_id(self) -> int:
        """Allocate a unique metadata node ID."""
        mid = self.next_metadata_id
        self.next_metadata_id += 1
        return mid

    def get_string_ident(self, val: str) -> str:
        """Get or create a global string constant identifier for a literal string."""
        if val in self.string_constants:
            return self.string_constants[val]
        name = f".str.{self.next_str_id}"
        self.next_str_id += 1
        self.string_constants[val] = name
        return name