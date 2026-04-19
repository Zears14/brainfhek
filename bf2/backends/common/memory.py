from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from bf2.core import ast as A
from bf2.core.errors import BF2RuntimeError


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


def _arr_type(et: A.TypeRef) -> Optional[str]:
    # We use lists for everything in the interpreter to avoid signedness issues with array.array
    return None


@dataclass
class Segment:
    name: str
    elem_type: A.TypeRef
    length: int
    struct_layout: Optional[StructLayout] = None
    cells: Union[array, list[Any]] = field(default_factory=list)

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

    def read_slot(self, seg: str, slot: int) -> Any:
        try:
            return self.segments[seg].cells[slot]
        except IndexError:
            length = len(self.segments[seg].cells)
            raise BF2RuntimeError(f"Index out of bounds reading segment '{seg}': slot {slot}, length {length}")

    def write_slot(self, seg: str, slot: int, val: Any) -> None:
        try:
            self.segments[seg].cells[slot] = val
        except IndexError:
            length = len(self.segments[seg].cells)
            raise BF2RuntimeError(f"Index out of bounds writing segment '{seg}': slot {slot}, length {length}")

    def resolve_ref(self, parts: list[Any]) -> tuple[str, int]:
        if not parts or parts[0] == "@":
            raise KeyError("bad ref")
        
        seg = str(parts[0])
        if seg not in self.segments:
            raise KeyError(f"unknown segment '{seg}'")
            
        s = self.segments[seg]
        layout = s.struct_layout
        slot = 0
        stride = layout.size if layout else 1
        i = 1
        while i < len(parts):
            p = parts[i]
            if isinstance(p, A.IntLit):
                idx = int(p.value)
                slot += idx * stride
                i += 1
                continue
            if isinstance(p, str) and p != "@":
                if layout is None:
                    raise KeyError(f"field '{p}' without struct layout")
                if p not in layout.offsets:
                    raise KeyError(f"unknown field '{p}' in struct '{layout.name}'")
                slot += layout.offsets[p]
                i += 1
                continue
            raise TypeError(f"Invalid reference part: {type(p)}")
        return seg, slot


def watch_key(seg: str, slot: int) -> str:
    return f"{seg}:{slot}"
