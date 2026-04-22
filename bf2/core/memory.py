from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

from bf2.core import ast as A


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
    fields: List[Tuple[str, A.TypeRef]]
    offsets: Dict[str, int]
    size: int


def build_struct_layout(decl: A.StructDecl) -> StructLayout:
    off = 0
    offs: Dict[str, int] = {}
    for fn, ft in decl.fields:
        offs[fn] = off
        off += max(1, type_size(ft))
    return StructLayout(decl.name, list(decl.fields), offs, off)
