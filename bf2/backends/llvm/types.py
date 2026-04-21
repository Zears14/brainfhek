"""Shared LLVM type mapping utilities for the BF2 backend."""

from __future__ import annotations

from llvmlite import ir

from bf2.core import ast as A

_TYPE_CACHE: dict = {}
_STRUCT_FIELDS: dict = {}
_IDENTIFIED_STRUCTS: dict = {}

Int1 = ir.IntType(1)
Int8 = ir.IntType(8)
Int16 = ir.IntType(16)
Int32 = ir.IntType(32)
Int64 = ir.IntType(64)
Float = ir.FloatType()
Double = ir.DoubleType()
Pointer = ir.PointerType(ir.IntType(8))


def _type_cache_key(t: A.TypeRef | None) -> tuple | None:
    if t is None:
        return None
    return (t.name, _type_cache_key(t.inner))


def to_ir_type(t: A.TypeRef) -> ir.Type:
    """Map a BF2 TypeRef to an llvmlite ir.Type."""
    key = _type_cache_key(t)
    if key in _TYPE_CACHE:
        return _TYPE_CACHE[key]

    if t.name == "ptr":
        pointee = to_ir_type(t.inner or A.TypeRef("i8"))
        result = ir.PointerType(pointee)
    elif t.name == "i8":
        result = Int8
    elif t.name == "bool":
        result = Int1
    elif t.name == "i16":
        result = Int16
    elif t.name == "i32":
        result = Int32
    elif t.name == "i64":
        result = Int64
    elif t.name == "f32":
        result = Float
    elif t.name == "f64":
        result = Double
    elif t.name == "void":
        result = ir.VoidType()
    elif t.name in _IDENTIFIED_STRUCTS:
        result = _IDENTIFIED_STRUCTS[t.name]
    else:
        result = Int32

    _TYPE_CACHE[key] = result
    return result


def get_struct_type(name: str, fields: list[A.TypeRef], module: ir.Module) -> ir.Type:
    """Get or create an identified struct type for a named struct declaration."""
    if name in _IDENTIFIED_STRUCTS:
        return _IDENTIFIED_STRUCTS[name]

    context = module.context if module else ir.Context()
    field_types = [to_ir_type(f) for f in fields]
    identified = context.get_identified_type(name)
    identified.set_body(*field_types)
    _IDENTIFIED_STRUCTS[name] = identified
    return identified


def type_str(t: ir.Type) -> str:
    """Convert an llvmlite ir.Type to its string representation."""
    return str(t)


def scalar_ty(t: A.TypeRef) -> str:
    """Map a BF2 TypeRef to an LLVM IR scalar type string (legacy compatibility)."""
    if t.name == "ptr":
        return "ptr"
    if t.name == "i8":
        return "i8"
    if t.name == "bool":
        return "i1"
    if t.name == "i16":
        return "i16"
    if t.name == "i32":
        return "i32"
    if t.name == "i64":
        return "i64"
    if t.name == "f32":
        return "float"
    if t.name == "f64":
        return "double"
    if t.name == "void":
        return "void"
    return f"%struct.{t.name}"


def align(t: str | ir.Type) -> int:
    """Return the alignment in bytes for an LLVM IR type."""
    if isinstance(t, ir.Type):
        t = str(t)

    if t == "i8" or t == "i1":
        return 1
    if t == "i16":
        return 2
    if t == "i32" or t == "float":
        return 4
    if t == "i64" or t == "double" or t == "ptr":
        return 8
    return 8


LINUX_LIBC = {
    "write": ("i64", ["i32", "ptr", "i64"]),
    "read": ("i64", ["i32", "ptr", "i64"]),
    "open": ("i32", ["ptr", "i32"]),
    "close": ("i32", ["i32"]),
    "exit": ("void", ["i32"]),
    "fork": ("i32", []),
    "getpid": ("i32", []),
    "strlen": ("i64", ["ptr"]),
    "nanosleep": ("i32", ["ptr", "ptr"]),
    "printf": ("i32", ["ptr", "..."]),
}
