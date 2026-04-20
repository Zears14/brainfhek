"""Shared LLVM type mapping utilities for the BF2 backend."""

from __future__ import annotations

from bf2.core import ast as A


def scalar_ty(t: A.TypeRef) -> str:
    """Map a BF2 TypeRef to an LLVM IR scalar type string."""
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
    return f"%struct.{t.name}" if t.name not in ("void", "") else "void"


def align(t: str) -> int:
    """Return the alignment in bytes for an LLVM IR type string."""
    if t == "i8":
        return 1
    if t == "i1" or t == "bool":
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
