from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any, Optional

from bf2 import ast_nodes as A


def type_size(t: A.TypeRef) -> int:
    if t.name == "ptr":
        return 8
    m = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8, "bool": 1}
    if t.name in m:
        return m[t.name]
    return 8


@dataclass
class StructLayout:
    name: str
    fields: list[tuple[str, A.TypeRef]]
    offsets: dict[str, int]
    size: int


def build_struct_layout(decl: A.StructDecl) -> StructLayout:
    off = 0
    offs: dict[str, int] = {}
    for fn, ft in decl.fields:
        offs[fn] = off
        off += max(1, type_size(ft))
    return StructLayout(decl.name, list(decl.fields), offs, off)


def _arr_type(et: A.TypeRef) -> str:
    m = {"i8": "b", "i16": "h", "i32": "i", "i64": "q", "f32": "f", "f64": "d", "bool": "B"}
    if et.name in m:
        return m[et.name]
    return "i"


@dataclass
class Segment:
    name: str
    elem_type: A.TypeRef
    length: int
    struct_layout: Optional[StructLayout] = None
    cells: array | list[int | float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cells:
            return
        el_slots = self.struct_layout.size if self.struct_layout else 1
        n = self.length * el_slots
        code = _arr_type(self.elem_type)
        try:
            self.cells = array(code, [0] * n)
        except TypeError:
            self.cells = [0] * n


@dataclass
class Pointer:
    seg: str
    slot: int
    pointee: A.TypeRef


class SegmentTable:
    def __init__(self, structs: dict[str, StructLayout]):
        self.structs = structs
        self.segments: dict[str, Segment] = {}

    def add_segment(self, decl: A.SegmentDecl, layout: Optional[StructLayout]) -> None:
        self.segments[decl.name] = Segment(decl.name, decl.elem_type, decl.length, layout, [])

    def read_slot(self, seg: str, slot: int) -> int | float:
        return self.segments[seg].cells[slot]

    def write_slot(self, seg: str, slot: int, val: int | float) -> None:
        self.segments[seg].cells[slot] = val

    def resolve_ref(self, parts: list[Any]) -> tuple[str, int]:
        if not parts or parts[0] == "@":
            raise KeyError("bad ref")
        from bf2.ast_nodes import IntLit

        seg = str(parts[0])
        s = self.segments[seg]
        layout = s.struct_layout
        slot = 0
        stride = layout.size if layout else 1
        i = 1
        while i < len(parts):
            p = parts[i]
            if isinstance(p, IntLit):
                idx = int(p.value)
                slot += idx * stride if layout else idx
                i += 1
                continue
            if isinstance(p, str) and p != "@":
                if layout is None:
                    raise KeyError("field without struct")
                slot += layout.offsets[p]
                i += 1
                continue
            raise TypeError(p)
        return seg, slot


def watch_key(seg: str, slot: int) -> str:
    return f"{seg}:{slot}"
