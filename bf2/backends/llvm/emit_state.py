"""Shared emission state threaded through all LLVM codegen sub-modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from bf2.core import ast as A
from bf2.backends.llvm.context import LLVMContext


@dataclass
class EmitState:
    """Mutable state container for a single LLVM IR emission pass.

    Replaces the loose ``self.*`` attributes previously scattered across
    ``LLVMEmitterVisitor``.  Every ``emit_*`` sub-module receives this
    object as its first argument.
    """

    mod: A.Module
    target: str
    ctx: LLVMContext
    lines: List[str] = field(default_factory=list)
    alloca_lines: List[str] = field(default_factory=list)

    # Collected top-level declarations
    structs: Dict[str, A.StructDecl] = field(default_factory=dict)
    fns: Dict[str, A.FunctionDef] = field(default_factory=dict)
    global_segs: Dict[str, A.SegmentDecl] = field(default_factory=dict)
    seg_slots: Dict[str, str] = field(default_factory=dict)
    watches: List[Tuple[str, int, A.ReactorDef]] = field(default_factory=list)

    # Metadata nodes collected during emission, appended at the end
    loop_metadata: List[str] = field(default_factory=list)
    next_metadata_id: int = 0

    # Function attribute groups
    fn_attrs: Dict[str, str] = field(default_factory=dict)

    # String constants for literals
    string_constants: Dict[str, str] = field(default_factory=dict)
    next_str_id: int = 0

    def alloc_metadata_id(self) -> int:
        """Allocate a unique metadata node ID."""
        mid = self.next_metadata_id
        self.next_metadata_id += 1
        return mid

    def get_string_ident(self, val: str) -> str:
        """Get or create a global string constant identifier for a literal string."""
        if val in self.string_constants:
            return self.string_constants[val]
        name = f"@.str.{self.next_str_id}"
        self.next_str_id += 1
        self.string_constants[val] = name
        return name
