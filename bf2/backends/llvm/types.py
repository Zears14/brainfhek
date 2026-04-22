"""Shared LLVM type mapping utilities for the BF2 backend."""

from __future__ import annotations

from llvmlite import ir

from bf2.core import ast as A

_STRUCT_FIELDS: dict = {}
_IDENTIFIED_STRUCTS: dict = {}


def clear_type_caches():
    """Clear global type caches to prevent pollution across compilations."""
    _STRUCT_FIELDS.clear()
    _IDENTIFIED_STRUCTS.clear()

Int1 = ir.IntType(1)
Int8 = ir.IntType(8)
Int16 = ir.IntType(16)
Int32 = ir.IntType(32)
Int64 = ir.IntType(64)
Float = ir.FloatType()
Double = ir.DoubleType()
Pointer = ir.PointerType(ir.IntType(8))




def to_ir_type(t: A.TypeRef) -> ir.Type:
    """Map a BF2 TypeRef to an llvmlite ir.Type."""
    if t.name == "ptr":
        pointee = to_ir_type(t.inner or A.TypeRef("i8"))
        return ir.PointerType(pointee)
    if t.name == "i8":
        return Int8
    if t.name == "bool":
        return Int1
    if t.name == "i16":
        return Int16
    if t.name == "i32":
        return Int32
    if t.name == "i64":
        return Int64
    if t.name == "f32":
        return Float
    if t.name == "f64":
        return Double
    if t.name == "void":
        return ir.VoidType()
    if t.name in _IDENTIFIED_STRUCTS:
        return _IDENTIFIED_STRUCTS[t.name]
    return Int32


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
