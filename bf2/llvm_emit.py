r"""
Emit standalone LLVM IR. With ``#include "stdlib"``, I/O uses ``write`` / ``read`` /
``snprintf`` (no ``printf`` / ``putchar``). Otherwise legacy lowering uses ``printf`` /
``putchar`` / ``getchar``. ``malloc`` / ``free`` and BF2 ``watch`` helpers are always
emitted as needed.

``clang out.ll -o a.out`` links libc.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from bf2 import ast_nodes as A
from bf2.ast_nodes import IntLit
from bf2.errors import SourceLoc
from bf2.memory import StructLayout, build_struct_layout, type_size as mem_type_size


class LLVMGenError(Exception):
    def __init__(self, msg: str, loc: SourceLoc | None = None):
        super().__init__(msg)
        self.loc = loc


def _struct_field_idx(layout: StructLayout, fname: str) -> int:
    for i, (fn, _) in enumerate(layout.fields):
        if fn == fname:
            return i
    raise LLVMGenError(f"unknown struct field {fname!r}")


def emit_llvm_ir(mod: A.Module, target: str | None = None) -> str:
    return _Emitter(mod, target).emit()


def _scalar_ty(t: A.TypeRef) -> str:
    if t.name == "ptr" and t.inner:
        return "ptr"
    m = {"i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64", "f32": "float", "f64": "double", "bool": "i1"}
    if t.name in m:
        return m[t.name]
    return f"%struct.{t.name}"


def _align(lty: str) -> int:
    if lty in ("i8", "i1"):
        return 1
    if lty in ("i16",):
        return 2
    if lty in ("i32", "float"):
        return 4
    return 8


_LINUX_LIBC_NAMES = frozenset(
    {
        "write",
        "read",
        "open",
        "close",
        "exit",
        "fork",
        "getpid",
        "strlen",
        "nanosleep",
    }
)


class _Ctx:
    __slots__ = (
        "ret_ty",
        "locals",
        "cursor_slot_ptr",
        "cursor_seg",
        "uid",
        "lid",
        "ptr_struct",
        "local_segs",
        "local_seg_alloca",
        "local_structs",
        "label_map",
        "pending_labels",
        "ptr_inner",
        "static_cslot",
    )

    def __init__(self, ret_ty: str) -> None:
        self.ret_ty = ret_ty
        self.locals: dict[str, tuple[str, str]] = {}
        self.cursor_slot_ptr = ""
        self.cursor_seg = "__bf"
        self.uid = 0
        self.lid = 0
        self.ptr_struct: dict[str, str] = {}
        self.local_segs: dict[str, A.SegmentDecl] = {}
        self.local_seg_alloca: dict[str, str] = {}
        self.local_structs: dict[str, A.StructDecl] = {}
        self.label_map: dict[str, str] = {}
        self.pending_labels: list[str] = []
        self.ptr_inner: dict[str, A.TypeRef] = {}
        self.static_cslot: int | None = 0

    def tmp(self) -> str:
        self.uid += 1
        return f"%t{self.uid}"

    def lab(self, h: str) -> str:
        self.lid += 1
        return f"{h}.{self.lid}"


class _Emitter:
    def __init__(self, mod: A.Module, target: str | None = None) -> None:
        self.mod = mod
        self.target = target
        self.lines: list[str] = []
        self.structs: dict[str, A.StructDecl] = {}
        self.layouts: dict[str, StructLayout] = {}
        self.segs: dict[str, A.SegmentDecl] = {}
        self.fns: dict[str, A.FunctionDef] = {}
        self.watches: list[tuple[A.RefExpr, A.Block, SourceLoc]] = []

    def emit(self) -> str:
        self._collect()
        self._header()
        self._declares()
        self._rx_helpers()
        self._struct_types()
        self._globals()
        for name in sorted(self.fns.keys(), key=lambda n: (n != "main", n)):
            self._emit_user_fn(self.fns[name], "main" if name == "main" else name)
        for i, (tgt, body, loc) in enumerate(self.watches):
            self._emit_watch_fn(i, tgt, body, loc)
        if "main" not in self.fns:
            raise LLVMGenError("module needs fn main")
        self.lines.append("attributes #0 = { nounwind }")
        return "\n".join(self.lines) + "\n"

    def _collect(self) -> None:
        for it in self.mod.items:
            if isinstance(it, A.StructDecl):
                self.structs[it.name] = it
                self.layouts[it.name] = build_struct_layout(it)
            elif isinstance(it, A.SegmentDecl):
                self.segs[it.name] = it
            elif isinstance(it, A.FunctionDef):
                self.fns[it.name] = it
                self._gather_structs_from_block(it.body)
            elif isinstance(it, A.ReactorDef):
                self.watches.append((it.target, it.body, it.loc))
                self._gather_structs_from_block(it.body)

    def _gather_structs_from_block(self, b: A.Block) -> None:
        for st in b.stmts:
            if isinstance(st, A.StructStmt):
                self.structs[st.decl.name] = st.decl
                self.layouts[st.decl.name] = build_struct_layout(st.decl)
            elif isinstance(st, A.IfStmt):
                self._gather_structs_from_block(st.then)
                if st.els:
                    self._gather_structs_from_block(st.els)
            elif isinstance(st, (A.LoopBF, A.LoopCounted)):
                self._gather_structs_from_block(st.body)

    def _header(self) -> None:
        self.lines.append('; ModuleID = "bf2"')
        self.lines.append('source_filename = "bf2"')
        self.lines.append(
            'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"'
        )
        tgt = self.target or "x86_64-pc-linux-gnu"
        self.lines.append(f'target triple = "{tgt}"')
        self.lines.append("")

    def _declares(self) -> None:
        ls = getattr(self.mod, "use_linux_stdlib", False)
        if not ls:
            self.lines.append("declare i32 @printf(ptr noundef, ...)")
            self.lines.append("declare i32 @putchar(i32)")
        self.lines.append("declare double @llvm.sqrt.f64(double)")
        self.lines.append("declare ptr @malloc(i64)")
        self.lines.append("declare void @free(ptr)")
        if not ls:
            self.lines.append("declare i32 @getchar()")
        self.lines.append("")
        if ls:
            self._linux_stdlib_declares()

    def _linux_stdlib_declares(self) -> None:
        self.lines.append("declare i32 @snprintf(ptr noundef, i64 noundef, ptr noundef, ...)")
        self.lines.append("declare i64 @write(i32 noundef, ptr noundef, i64 noundef)")
        self.lines.append("declare i64 @read(i32 noundef, ptr noundef, i64 noundef)")
        self.lines.append("declare i32 @open(ptr noundef, i32 noundef, ...)")
        self.lines.append("declare i32 @close(i32 noundef)")
        self.lines.append("declare void @exit(i32 noundef)")
        self.lines.append("declare i32 @fork()")
        self.lines.append("declare i32 @getpid()")
        self.lines.append("declare i64 @strlen(ptr noundef)")
        self.lines.append("declare i32 @nanosleep(ptr noundef, ptr noundef)")
        self.lines.append("")

    def _rx_helpers(self) -> None:
        self.lines.append("@bf2.watch.depth = global i32 0")
        self.lines.append("@bf2.rx.depth = global i32 0")
        self.lines.append("define void @bf2.rx.begin() #0 {")
        self.lines.append("entry:")
        self.lines.append("  %d = load i32, ptr @bf2.rx.depth, align 4")
        self.lines.append("  %n = add i32 %d, 1")
        self.lines.append("  store i32 %n, ptr @bf2.rx.depth, align 4")
        self.lines.append("  ret void")
        self.lines.append("}")
        self.lines.append("define void @bf2.rx.end() #0 {")
        self.lines.append("entry:")
        self.lines.append("  %d = load i32, ptr @bf2.rx.depth, align 4")
        self.lines.append("  %c = icmp sgt i32 %d, 0")
        self.lines.append("  br i1 %c, label %dec, label %join")
        self.lines.append("dec:")
        self.lines.append("  %m = sub i32 %d, 1")
        self.lines.append("  store i32 %m, ptr @bf2.rx.depth, align 4")
        self.lines.append("  br label %join")
        self.lines.append("join:")
        self.lines.append("  ret void")
        self.lines.append("}")
        self.lines.append("define i1 @bf2.rx.suppress() #0 {")
        self.lines.append("entry:")
        self.lines.append("  %d = load i32, ptr @bf2.rx.depth, align 4")
        self.lines.append("  %s = icmp sgt i32 %d, 1")
        self.lines.append("  ret i1 %s")
        self.lines.append("}")
        self.lines.append("")

    def _struct_types(self) -> None:
        if getattr(self.mod, "use_linux_stdlib", False):
            self.lines.append("%struct.timespec = type { i64, i64 }")
        for name, decl in self.structs.items():
            parts = ", ".join(_scalar_ty(ft) for _, ft in decl.fields)
            self.lines.append(f"%struct.{name} = type {{ {parts} }}")

    def _seg_arr_ty(self, decl: A.SegmentDecl) -> str:
        et = decl.elem_type
        inner = f"%struct.{et.name}" if et.name in self.structs else _scalar_ty(et)
        return f"[{decl.length} x {inner}]"

    def _globals(self) -> None:
        self.lines.append("")
        self.lines.append("@__bf = global [30000 x i8] zeroinitializer, align 16")
        self.lines.append("@__cp = global [1 x i32] zeroinitializer, align 4")
        for name, decl in self.segs.items():
            aty = self._seg_arr_ty(decl)
            self.lines.append(f"@{name} = global {aty} zeroinitializer, align 16")
        self.lines.append('@fmt.dnl = private unnamed_addr constant [4 x i8] c"%d\\0A\\00", align 1')
        self.lines.append('@fmt.d = private unnamed_addr constant [3 x i8] c"%d\\00", align 1')
        self.lines.append('@fmt.g = private unnamed_addr constant [7 x i8] c"%.17g\\0A\\00", align 1')
        if getattr(self.mod, "use_linux_stdlib", False):
            self.lines.append("@STDIN = global i32 0")
            self.lines.append("@STDOUT = global i32 1")
            self.lines.append("@STDERR = global i32 2")
            self.lines.append("@O_RDONLY = global i32 0")
            self.lines.append("@O_CREAT = global i32 64")
        self.lines.append("")

    def _decl_for_seg(self, name: str, ctx: _Ctx) -> A.SegmentDecl:
        if name == "__bf":
            return A.SegmentDecl("__bf", A.TypeRef("i8"), 30000, SourceLoc(1, 1))
        if name == "__cp":
            return A.SegmentDecl("__cp", A.TypeRef("i32"), 1, SourceLoc(1, 1))
        d = ctx.local_segs.get(name) or self.segs.get(name)
        if not d:
            raise LLVMGenError(f"unknown segment {name}")
        return d

    def _seg_base_sym(self, name: str, ctx: _Ctx) -> str:
        if name in ctx.local_seg_alloca:
            return ctx.local_seg_alloca[name]
        if name == "__bf":
            return "@__bf"
        if name == "__cp":
            return "@__cp"
        if name in self.segs:
            return f"@{name}"
        raise LLVMGenError(f"unknown segment {name}")

    def _arr_ty_for_cursor_seg(self, ctx: _Ctx) -> Tuple[str, str]:
        d = self._decl_for_seg(ctx.cursor_seg, ctx)
        aty = self._seg_arr_ty(d)
        et = d.elem_type
        stn = et.name
        inner = f"%struct.{stn}" if stn in self.structs else _scalar_ty(et)
        return aty, inner

    def _cursor_gep(self, ctx: _Ctx) -> Tuple[str, str]:
        aty, inner = self._arr_ty_for_cursor_seg(ctx)
        base = self._seg_base_sym(ctx.cursor_seg, ctx)
        if ctx.static_cslot is not None:
            si = str(ctx.static_cslot)
        else:
            si = ctx.tmp()
            self.lines.append(f"  {si} = load i32, ptr {ctx.cursor_slot_ptr}, align 4")
        g = ctx.tmp()
        self.lines.append(f"  {g} = getelementptr inbounds {aty}, ptr {base}, i32 0, i32 {si}")
        return g, inner

    def _load_typed_as_i32(self, ptr: str, lty: str, ctx: _Ctx) -> str:
        al = _align(lty)
        t = ctx.tmp()
        self.lines.append(f"  {t} = load {lty}, ptr {ptr}, align {al}")
        if lty == "i32":
            return t
        if lty == "i8":
            z = ctx.tmp()
            self.lines.append(f"  {z} = zext i8 {t} to i32")
            return z
        if lty == "i16":
            z = ctx.tmp()
            self.lines.append(f"  {z} = sext i16 {t} to i32")
            return z
        if lty == "i64":
            z = ctx.tmp()
            self.lines.append(f"  {z} = trunc i64 {t} to i32")
            return z
        if lty == "double":
            z = ctx.tmp()
            self.lines.append(f"  {z} = fptosi double {t} to i32")
            return z
        if lty == "float":
            z = ctx.tmp()
            self.lines.append(f"  {z} = fptosi float {t} to i32")
            return z
        raise LLVMGenError(f"load cursor {lty}")

    def _store_i32_to_typed(self, ptr: str, lty: str, val: str, ctx: _Ctx) -> None:
        al = _align(lty)
        if lty == "i32":
            self.lines.append(f"  store i32 {val}, ptr {ptr}, align {al}")
        elif lty == "i8":
            t = ctx.tmp()
            self.lines.append(f"  {t} = trunc i32 {val} to i8")
            self.lines.append(f"  store i8 {t}, ptr {ptr}, align 1")
        elif lty == "i16":
            t = ctx.tmp()
            self.lines.append(f"  {t} = trunc i32 {val} to i16")
            self.lines.append(f"  store i16 {t}, ptr {ptr}, align {al}")
        elif lty == "i64":
            t = ctx.tmp()
            self.lines.append(f"  {t} = sext i32 {val} to i64")
            self.lines.append(f"  store i64 {t}, ptr {ptr}, align {al}")
        elif lty == "double":
            t = ctx.tmp()
            self.lines.append(f"  {t} = sitofp i32 {val} to double")
            self.lines.append(f"  store double {t}, ptr {ptr}, align {al}")
        elif lty == "float":
            t = ctx.tmp()
            self.lines.append(f"  {t} = sitofp i32 {val} to float")
            self.lines.append(f"  store float {t}, ptr {ptr}, align {al}")
        else:
            raise LLVMGenError(f"store cell {lty}")

    def _cursor_val_i32(self, ctx: _Ctx) -> str:
        ptr, lty = self._cursor_gep(ctx)
        return self._load_typed_as_i32(ptr, lty, ctx)

    def _try_static_seg_slot(self, r: A.RefExpr, ctx: _Ctx) -> Optional[tuple[str, int]]:
        p = r.parts
        if not p or p[0] == "@":
            return None
        head = str(p[0])
        if head in ctx.ptr_struct:
            return None
        decl = ctx.local_segs.get(head) or self.segs.get(head)
        if not decl:
            return None
        layout = self.layouts.get(decl.elem_type.name) if decl.elem_type.name in self.layouts else None
        slot = 0
        stride = layout.size if layout else 1
        i = 1
        while i < len(p):
            x = p[i]
            if isinstance(x, IntLit):
                idx = int(x.value)
                slot += idx * stride if layout else idx
                i += 1
                continue
            if isinstance(x, str) and layout:
                slot += layout.offsets[x]
                i += 1
                continue
            return None
        return (head, slot)

    def _elem_byte_size(self, t: A.TypeRef) -> int:
        if t.name in self.layouts:
            return self.layouts[t.name].size
        return mem_type_size(t)

    def _emit_dynamic_slot(self, r: A.RefExpr, ctx: _Ctx) -> tuple[str, str]:
        p = r.parts
        if not p or p[0] == "@":
            raise LLVMGenError("bad ref", r.loc)
        head = str(p[0])
        if head in ctx.ptr_struct:
            raise LLVMGenError("MoveOp through ptr not supported for LLVM", r.loc)
        decl = ctx.local_segs.get(head) or self.segs.get(head)
        if not decl:
            raise LLVMGenError(f"unknown segment {head}", r.loc)
        layout = self.layouts.get(decl.elem_type.name) if decl.elem_type.name in self.layouts else None
        stride = layout.size if layout else 1
        acc: str | None = None
        i = 1
        while i < len(p):
            x = p[i]
            if isinstance(x, IntLit):
                c = int(x.value) * stride if layout else int(x.value)
                if acc is None:
                    acc = str(c)
                else:
                    t = ctx.tmp()
                    self.lines.append(f"  {t} = add nsw i32 {acc}, {c}")
                    acc = t
            elif isinstance(x, str) and layout:
                c = layout.offsets[x]
                if acc is None:
                    acc = str(c)
                else:
                    t = ctx.tmp()
                    self.lines.append(f"  {t} = add nsw i32 {acc}, {c}")
                    acc = t
            else:
                ex, et = self._emit_expr(x, ctx)
                ev = ex
                if et == "double":
                    ev = ctx.tmp()
                    self.lines.append(f"  {ev} = fptosi double {ex} to i32")
                elif et != "i32":
                    raise LLVMGenError("index must be i32", r.loc)
                term = ev
                if stride != 1:
                    t2 = ctx.tmp()
                    self.lines.append(f"  {t2} = mul nsw i32 {ev}, {stride}")
                    term = t2
                if acc is None:
                    acc = term
                else:
                    t3 = ctx.tmp()
                    self.lines.append(f"  {t3} = add nsw i32 {acc}, {term}")
                    acc = t3
            i += 1
        if acc is None:
            acc = "0"
        return head, acc

    def _store_typed_at_ptr(self, ptr: str, lty: str, rhs: str, rty: str, ctx: _Ctx) -> None:
        al = _align(lty)
        if lty == "double" and rty == "i32":
            d = ctx.tmp()
            self.lines.append(f"  {d} = sitofp i32 {rhs} to double")
            self.lines.append(f"  store double {d}, ptr {ptr}, align {al}")
            return
        if rty == "i32" and lty in ("i8", "i16"):
            t = ctx.tmp()
            self.lines.append(f"  {t} = trunc i32 {rhs} to {lty}")
            self.lines.append(f"  store {lty} {t}, ptr {ptr}, align {al}")
            return
        if lty == rty:
            self.lines.append(f"  store {lty} {rhs}, ptr {ptr}, align {al}")
            return
        raise LLVMGenError(f"store {lty} <- {rty}")

    def _load_stdio_fd(self, sym: str, ctx: _Ctx) -> str:
        t = ctx.tmp()
        self.lines.append(f"  {t} = load i32, ptr {sym}, align 4")
        return t

    def _emit_linux_snprintf_write(
        self, ctx: _Ctx, val_ssa: str, val_llvm_ty: str, fmt_global: str
    ) -> None:
        buf = ctx.tmp()
        self.lines.append(f"  {buf} = alloca [64 x i8], align 16")
        p = ctx.tmp()
        self.lines.append(f"  {p} = getelementptr inbounds [64 x i8], ptr {buf}, i32 0, i32 0")
        if val_llvm_ty == "i32":
            self.lines.append(
                f"  call i32 @snprintf(ptr {p}, i64 64, ptr {fmt_global}, i32 {val_ssa})"
            )
        elif val_llvm_ty == "double":
            self.lines.append(
                f"  call i32 @snprintf(ptr {p}, i64 64, ptr {fmt_global}, double {val_ssa})"
            )
        else:
            raise LLVMGenError(f"snprintf val {val_llvm_ty}")
        ln = ctx.tmp()
        self.lines.append(f"  {ln} = call i64 @strlen(ptr {p})")
        fd = self._load_stdio_fd("@STDOUT", ctx)
        w = ctx.tmp()
        self.lines.append(f"  {w} = call i64 @write(i32 {fd}, ptr {p}, i64 {ln})")

    def _emit_io_linux(self, st: A.IOStmt, ctx: _Ctx) -> None:
        if st.kind == ".i" and st.expr:
            v, et = self._emit_expr(st.expr, ctx)
            if et == "double":
                vv = ctx.tmp()
                self.lines.append(f"  {vv} = fptosi double {v} to i32")
                v = vv
            self._emit_linux_snprintf_write(ctx, v, "i32", "@fmt.dnl")
        elif st.kind == ".i":
            v = self._cursor_val_i32(ctx)
            self._emit_linux_snprintf_write(ctx, v, "i32", "@fmt.dnl")
        elif st.kind == ".f" and st.expr:
            v, et = self._emit_expr(st.expr, ctx)
            if et == "i32":
                dv = ctx.tmp()
                self.lines.append(f"  {dv} = sitofp i32 {v} to double")
                v = dv
            self._emit_linux_snprintf_write(ctx, v, "double", "@fmt.g")
        elif st.kind == ".":
            v = self._cursor_val_i32(ctx)
            ch = ctx.tmp()
            self.lines.append(f"  {ch} = and i32 {v}, 255")
            b = ctx.tmp()
            self.lines.append(f"  {b} = alloca i8, align 1")
            t8 = ctx.tmp()
            self.lines.append(f"  {t8} = trunc i32 {ch} to i8")
            self.lines.append(f"  store i8 {t8}, ptr {b}, align 1")
            fd = self._load_stdio_fd("@STDOUT", ctx)
            w = ctx.tmp()
            self.lines.append(f"  {w} = call i64 @write(i32 {fd}, ptr {b}, i64 1)")
        elif st.kind == ",":
            b = ctx.tmp()
            self.lines.append(f"  {b} = alloca i8, align 1")
            fdin = self._load_stdio_fd("@STDIN", ctx)
            self.lines.append(f"  call i64 @read(i32 {fdin}, ptr {b}, i64 1)")
            t8 = ctx.tmp()
            self.lines.append(f"  {t8} = load i8, ptr {b}, align 1")
            z = ctx.tmp()
            self.lines.append(f"  {z} = zext i8 {t8} to i32")
            dptr, dty = self._cursor_gep(ctx)
            self._store_i32_to_typed(dptr, dty, z, ctx)
        else:
            raise LLVMGenError(f"io {st.kind}", st.loc)

    def _emit_maybe_watch(self, ctx: _Ctx, seg: str, slot: int) -> None:
        if not self.watches:
            return
        key = f"{seg}:{slot}"
        for wi, (tgt, _, _) in enumerate(self.watches):
            sk = self._try_static_seg_slot(tgt, ctx)
            if sk and f"{sk[0]}:{sk[1]}" == key:
                d = ctx.tmp()
                self.lines.append(f"  {d} = load i32, ptr @bf2.watch.depth, align 4")
                ns = ctx.tmp()
                self.lines.append(f"  {ns} = icmp sgt i32 {d}, 0")
                skip = ctx.lab("wx")
                doo = ctx.lab("wx")
                join = ctx.lab("wx")
                self.lines.append(f"  br i1 {ns}, label %{skip}, label %{doo}")
                self.lines.append(f"{doo}:")
                n1 = ctx.tmp()
                self.lines.append(f"  {n1} = add i32 {d}, 1")
                self.lines.append(f"  store i32 {n1}, ptr @bf2.watch.depth, align 4")
                self.lines.append(f"  call void @bf2.watch.{wi}()")
                d2 = ctx.tmp()
                self.lines.append(f"  {d2} = load i32, ptr @bf2.watch.depth, align 4")
                n2 = ctx.tmp()
                self.lines.append(f"  {n2} = sub i32 {d2}, 1")
                self.lines.append(f"  store i32 {n2}, ptr @bf2.watch.depth, align 4")
                self.lines.append(f"  br label %{join}")
                self.lines.append(f"{skip}:")
                self.lines.append(f"  br label %{join}")
                self.lines.append(f"{join}:")

    def _emit_watch_fn(self, idx: int, tgt: A.RefExpr, body: A.Block, loc: SourceLoc) -> None:
        sk = self._try_static_seg_slot(tgt, _Ctx("i32"))
        if not sk:
            raise LLVMGenError("watch target must resolve to a static segment slot for LLVM", loc)
        seg, slot_v = sk
        self.lines.append(f"define void @bf2.watch.{idx}() #0 {{")
        self.lines.append("entry:")
        ctx = _Ctx("void")
        ctx.cursor_slot_ptr = "%wcslot"
        self.lines.append("  %wcslot = alloca i32, align 4")
        self.lines.append(f"  store i32 {slot_v}, ptr %wcslot, align 4")
        ctx.cursor_seg = seg
        self._emit_block(body, ctx)
        self.lines.append("  ret void")
        self.lines.append("}")
        self.lines.append("")

    def _emit_user_fn(self, fd: A.FunctionDef, sym: str) -> None:
        ret = _scalar_ty(fd.ret)
        args = ", ".join(f"{_scalar_ty(t)} %{n}" for n, t in fd.params)
        self.lines.append(f"define {ret} @{sym}({args}) #0 {{")
        self.lines.append("entry:")
        ctx = _Ctx(ret)
        ctx.cursor_slot_ptr = "%cslot"
        self.lines.append("  %cslot = alloca i32, align 4")
        self.lines.append("  store i32 0, ptr %cslot, align 4")
        ctx.cursor_seg = "__bf"
        if fd.params and fd.params[0][1].name in ("i8", "i16", "i32", "i64"):
            pn, pt = fd.params[0]
            pty = _scalar_ty(pt)
            self.lines.append("  %cpg = getelementptr inbounds [1 x i32], ptr @__cp, i32 0, i32 0")
            if pt.name == "i32":
                self.lines.append(f"  store i32 %{pn}, ptr %cpg, align 4")
            elif pt.name == "i64":
                t0 = ctx.tmp()
                self.lines.append(f"  {t0} = trunc i64 %{pn} to i32")
                self.lines.append(f"  store i32 {t0}, ptr %cpg, align 4")
            else:
                self.lines.append(f"  %e0 = zext {pty} %{pn} to i32")
                self.lines.append("  store i32 %e0, ptr %cpg, align 4")
            self.lines.append("  store i32 0, ptr %cslot, align 4")
            ctx.cursor_seg = "__cp"
        for n, t in fd.params:
            p = ctx.tmp()
            if t.name == "ptr" and t.inner:
                self.lines.append(f"  {p} = alloca ptr, align 8")
                self.lines.append(f"  store ptr %{n}, ptr {p}, align 8")
                ctx.locals[n] = (p, "ptr")
                ctx.ptr_struct[n] = t.inner.name
                ctx.ptr_inner[n] = t.inner
            else:
                ty = _scalar_ty(t)
                al = _align(ty)
                self.lines.append(f"  {p} = alloca {ty}, align {al}")
                self.lines.append(f"  store {ty} %{n}, ptr {p}, align {al}")
                ctx.locals[n] = (p, ty)
        self._emit_block(fd.body, ctx)
        if not self._is_terminated():
            z = "0" if ret == "i32" else ("0.0" if ret == "double" else "0")
            self.lines.append(f"  ret {ret} {z}")
        self.lines.append("}")
        self.lines.append("")

    def _is_terminated(self) -> bool:
        if not self.lines:
            return False
        last = self.lines[-1].lstrip()
        return last.startswith("ret ") or last.startswith("br ")

    def _has_ret(self, b: A.Block) -> bool:
        return any(isinstance(s, A.RetStmt) for s in b.stmts)

    def _emit_block(self, b: A.Block, ctx: _Ctx) -> None:
        labs: dict[str, str] = {}
        for st in b.stmts:
            if isinstance(st, A.LabelStmt):
                labs[st.name] = ctx.lab("L")
        merged = {**ctx.label_map, **labs}
        o = ctx.label_map
        ctx.label_map = merged
        for st in b.stmts:
            if isinstance(st, A.LabelStmt):
                ctx.static_cslot = None
                self.lines.append(f"{labs[st.name]}:")
                continue
            if self._is_terminated():
                continue
            self._emit_stmt(st, ctx)
        ctx.label_map = o

    def _emit_stmt(self, st: Any, ctx: _Ctx) -> None:
        if isinstance(st, A.JumpStmt):
            tgt = ctx.label_map.get(st.name)
            if not tgt:
                raise LLVMGenError(f"unknown label {st.name}", st.loc)
            self.lines.append(f"  br label %{tgt}")
            return
        if isinstance(st, A.RetStmt):
            if st.value is None:
                self.lines.append(f"  ret {_scalar_ty(A.TypeRef('i32'))} 0" if ctx.ret_ty == "i32" else "  ret void")
            else:
                v, ty = self._emit_expr(st.value, ctx)
                if ctx.ret_ty == "i32" and ty == "double":
                    rv = ctx.tmp()
                    self.lines.append(f"  {rv} = fptosi double {v} to i32")
                    self.lines.append(f"  ret i32 {rv}")
                else:
                    self.lines.append(f"  ret {ty} {v}")
            return
        if isinstance(st, A.PtrDecl):
            p = ctx.tmp()
            self.lines.append(f"  {p} = alloca ptr, align 8")
            ctx.ptr_struct[st.name] = st.inner.name
            ctx.ptr_inner[st.name] = st.inner
            v, _ = self._emit_expr(st.init, ctx)
            self.lines.append(f"  store ptr {v}, ptr {p}, align 8")
            ctx.locals[st.name] = (p, "ptr")
            return
        if isinstance(st, A.StructStmt):
            self.structs[st.decl.name] = st.decl
            self.layouts[st.decl.name] = build_struct_layout(st.decl)
            ctx.local_structs[st.decl.name] = st.decl
            return
        if isinstance(st, A.VarDecl):
            ty = _scalar_ty(st.ty)
            al = _align(ty)
            p = ctx.tmp()
            self.lines.append(f"  {p} = alloca {ty}, align {al}")
            if st.ty.name == "ptr" and st.ty.inner:
                ctx.ptr_struct[st.name] = st.ty.inner.name
                ctx.ptr_inner[st.name] = st.ty.inner
            if st.init:
                v, et = self._emit_expr(st.init, ctx)
                if ty == "double" and et == "i32":
                    vd = ctx.tmp()
                    self.lines.append(f"  {vd} = sitofp i32 {v} to double")
                    self.lines.append(f"  store double {vd}, ptr {p}, align {al}")
                else:
                    self.lines.append(f"  store {et} {v}, ptr {p}, align {al}")
            else:
                self.lines.append(f"  store {ty} zeroinitializer, ptr {p}, align {al}")
            ctx.locals[st.name] = (p, ty)
            return
        if isinstance(st, A.AssignStmt):
            rhs, rty = self._emit_expr(st.rhs, ctx)
            self._store_lhs(st.lhs, rhs, rty, ctx)
            return
        if isinstance(st, A.CallStmt):
            self._emit_call_expr(st.call, ctx, use_result=False)
            return
        if isinstance(st, A.IfStmt):
            cv = self._cursor_val_i32(ctx)
            cond = self._emit_cond(st.cond, cv, ctx)
            ctx.static_cslot = None
            Lthen = ctx.lab("then")
            Lelse = ctx.lab("else")
            Lend = ctx.lab("endif")
            if st.els:
                self.lines.append(f"  br i1 {cond}, label %{Lthen}, label %{Lelse}")
            else:
                self.lines.append(f"  br i1 {cond}, label %{Lthen}, label %{Lend}")
            self.lines.append(f"{Lthen}:")
            self._emit_block(st.then, ctx)
            if not self._is_terminated():
                self.lines.append(f"  br label %{Lend}")
            if st.els:
                self.lines.append(f"{Lelse}:")
                self._emit_block(st.els, ctx)
                if not self._is_terminated():
                    self.lines.append(f"  br label %{Lend}")
            self.lines.append(f"{Lend}:")
            return
        if isinstance(st, A.IOStmt):
            if getattr(self.mod, "use_linux_stdlib", False):
                self._emit_io_linux(st, ctx)
            elif st.kind == ".i" and st.expr:
                v, _ = self._emit_expr(st.expr, ctx)
                self.lines.append(f"  call i32 @printf(ptr @fmt.dnl, i32 {v})")
            elif st.kind == ".i":
                v = self._cursor_val_i32(ctx)
                self.lines.append(f"  call i32 @printf(ptr @fmt.dnl, i32 {v})")
            elif st.kind == ".f" and st.expr:
                v, _ = self._emit_expr(st.expr, ctx)
                self.lines.append(f"  call i32 @printf(ptr @fmt.g, double {v})")
            elif st.kind == ".":
                v = self._cursor_val_i32(ctx)
                ch = ctx.tmp()
                self.lines.append(f"  {ch} = and i32 {v}, 255")
                self.lines.append(f"  call i32 @putchar(i32 {ch})")
            elif st.kind == ",":
                gc = ctx.tmp()
                self.lines.append(f"  {gc} = call i32 @getchar()")
                dptr, dty = self._cursor_gep(ctx)
                self._store_i32_to_typed(dptr, dty, gc, ctx)
            else:
                raise LLVMGenError(f"io {st.kind}", st.loc)
            return
        if isinstance(st, A.LoadOp):
            sptr, sty = self._gep(st.src, ctx)
            lv = self._load_typed_as_i32(sptr, sty, ctx)
            dptr, dty = self._cursor_gep(ctx)
            self._store_i32_to_typed(dptr, dty, lv, ctx)
            return
        if isinstance(st, A.StoreOp):
            v = self._cursor_val_i32(ctx)
            ptr, ty = self._gep(st.dst, ctx)
            self._store_i32_to_typed(ptr, ty, v, ctx)
            sk = self._try_static_seg_slot(st.dst, ctx)
            if sk:
                self._emit_maybe_watch(ctx, sk[0], sk[1])
            return
        if isinstance(st, A.SwapOp):
            p0, t0 = self._cursor_gep(ctx)
            v0 = self._load_typed_as_i32(p0, t0, ctx)
            p1, t1 = self._gep(st.other, ctx)
            v1 = self._load_typed_as_i32(p1, t1, ctx)
            self._store_i32_to_typed(p0, t0, v1, ctx)
            self._store_i32_to_typed(p1, t1, v0, ctx)
            return
        if isinstance(st, A.MoveOp):
            if st.target.parts and st.target.parts[0] == "@":
                return
            sk = self._try_static_seg_slot(st.target, ctx)
            if sk:
                ctx.cursor_seg, sslot = sk
                ctx.static_cslot = sslot
                self.lines.append(f"  store i32 {sslot}, ptr {ctx.cursor_slot_ptr}, align 4")
                return
            ctx.static_cslot = None
            seg, slot_ssa = self._emit_dynamic_slot(st.target, ctx)
            ctx.cursor_seg = seg
            self.lines.append(f"  store i32 {slot_ssa}, ptr {ctx.cursor_slot_ptr}, align 4")
            return
        if isinstance(st, A.MoveRel):
            ctx.static_cslot = None
            cur = ctx.tmp()
            self.lines.append(f"  {cur} = load i32, ptr {ctx.cursor_slot_ptr}, align 4")
            n = ctx.tmp()
            self.lines.append(f"  {n} = add nsw i32 {cur}, {st.delta}")
            self.lines.append(f"  store i32 {n}, ptr {ctx.cursor_slot_ptr}, align 4")
            return
        if isinstance(st, A.CellAssignLit):
            ptr, lty = self._cursor_gep(ctx)
            if isinstance(st.value, bool):
                ivs = "1" if st.value else "0"
                self._store_i32_to_typed(ptr, lty, ivs, ctx)
            elif isinstance(st.value, float):
                t = ctx.tmp()
                self.lines.append(f"  {t} = fptosi double {float(st.value)} to i32")
                self._store_i32_to_typed(ptr, lty, t, ctx)
            else:
                self._store_i32_to_typed(ptr, lty, str(int(st.value)), ctx)
            return
        if isinstance(st, A.CellArith):
            p, lty = self._cursor_gep(ctx)
            v = self._load_typed_as_i32(p, lty, ctx)
            amt = st.amount
            if amt is None:
                amt = 1 if st.op in "+-" else 0
            imm = int(amt) if isinstance(amt, (int, float)) else 1
            r = ctx.tmp()
            if st.op == "+":
                self.lines.append(f"  {r} = add nsw i32 {v}, {imm}")
            elif st.op == "-":
                self.lines.append(f"  {r} = sub nsw i32 {v}, {imm}")
            elif st.op == "*":
                self.lines.append(f"  {r} = mul nsw i32 {v}, {imm}")
            elif st.op == "/":
                self.lines.append(f"  {r} = sdiv i32 {v}, {imm}")
            else:
                raise LLVMGenError("cell op", st.loc)
            self._store_i32_to_typed(p, lty, r, ctx)
            return
        if isinstance(st, A.CellArithRef):
            ptr, lty = self._gep(st.target, ctx)
            v = self._load_typed_as_i32(ptr, lty, ctx)
            amt = int(st.amount) if isinstance(st.amount, float) else int(st.amount)
            r = ctx.tmp()
            if st.op == "+":
                self.lines.append(f"  {r} = add nsw i32 {v}, {amt}")
            elif st.op == "-":
                self.lines.append(f"  {r} = sub nsw i32 {v}, {amt}")
            elif st.op == "*":
                self.lines.append(f"  {r} = mul nsw i32 {v}, {amt}")
            elif st.op == "/":
                self.lines.append(f"  {r} = sdiv i32 {v}, {amt}")
            else:
                raise LLVMGenError("cell ref op", st.loc)
            self._store_i32_to_typed(ptr, lty, r, ctx)
            sk = self._try_static_seg_slot(st.target, ctx)
            if sk:
                self._emit_maybe_watch(ctx, sk[0], sk[1])
            return
        if isinstance(st, A.LoopBF):
            ctx.static_cslot = None
            Lh, Lb, Le = ctx.lab("bfh"), ctx.lab("bfb"), ctx.lab("bfe")
            self.lines.append(f"  br label %{Lh}")
            self.lines.append(f"{Lh}:")
            vv = self._cursor_val_i32(ctx)
            c = ctx.tmp()
            self.lines.append(f"  {c} = icmp ne i32 {vv}, 0")
            self.lines.append(f"  br i1 {c}, label %{Lb}, label %{Le}")
            self.lines.append(f"{Lb}:")
            self._emit_block(st.body, ctx)
            if not self._is_terminated():
                self.lines.append(f"  br label %{Lh}")
            self.lines.append(f"{Le}:")
            return
        if isinstance(st, A.SegmentStmt):
            decl = st.decl
            aty = self._seg_arr_ty(decl)
            base = ctx.tmp()
            self.lines.append(f"  {base} = alloca {aty}, align 16")
            ctx.local_segs[decl.name] = decl
            ctx.local_seg_alloca[decl.name] = base
            return
        if isinstance(st, A.LoopCounted):
            ctx.static_cslot = None
            ctr = ctx.tmp()
            self.lines.append(f"  {ctr} = alloca i32, align 4")
            self.lines.append(f"  store i32 0, ptr {ctr}, align 4")
            Lp, Lb, Le = ctx.lab("clp"), ctx.lab("clb"), ctx.lab("cle")
            self.lines.append(f"  br label %{Lp}")
            self.lines.append(f"{Lp}:")
            ci = ctx.tmp()
            self.lines.append(f"  {ci} = load i32, ptr {ctr}, align 4")
            ck = ctx.tmp()
            self.lines.append(f"  {ck} = icmp slt i32 {ci}, {st.count}")
            self.lines.append(f"  br i1 {ck}, label %{Lb}, label %{Le}")
            self.lines.append(f"{Lb}:")
            self._emit_block(st.body, ctx)
            if not self._is_terminated():
                ci2 = ctx.tmp()
                self.lines.append(f"  {ci2} = load i32, ptr {ctr}, align 4")
                cn = ctx.tmp()
                self.lines.append(f"  {cn} = add nsw i32 {ci2}, 1")
                self.lines.append(f"  store i32 {cn}, ptr {ctr}, align 4")
                self.lines.append(f"  br label %{Lp}")
            self.lines.append(f"{Le}:")
            return
        if isinstance(st, A.AllocStmt):
            esz = self._elem_byte_size(st.ty)
            total = int(st.count) * esz
            raw = ctx.tmp()
            self.lines.append(f"  {raw} = call ptr @malloc(i64 {total})")
            if not st.name:
                return
            p = ctx.tmp()
            self.lines.append(f"  {p} = alloca ptr, align 8")
            self.lines.append(f"  store ptr {raw}, ptr {p}, align 8")
            ctx.locals[st.name] = (p, "ptr")
            ctx.ptr_inner[st.name] = st.ty
            if st.ty.name in self.structs:
                ctx.ptr_struct[st.name] = st.ty.name
            return
        if isinstance(st, A.ExprStmt):
            self._emit_expr(st.expr, ctx)
            return
        if isinstance(st, A.PtrArith):
            if st.name not in ctx.locals:
                raise LLVMGenError(f"unknown ptr {st.name}", st.loc)
            inn = ctx.ptr_inner.get(st.name)
            if not inn:
                raise LLVMGenError("ptr arith needs a ptr local", st.loc)
            lp, _ = ctx.locals[st.name]
            esz = self._elem_byte_size(inn)
            off = int(st.delta) * esz
            pb = ctx.tmp()
            self.lines.append(f"  {pb} = load ptr, ptr {lp}, align 8")
            nb = ctx.tmp()
            self.lines.append(f"  {nb} = getelementptr inbounds i8, ptr {pb}, i64 {off}")
            self.lines.append(f"  store ptr {nb}, ptr {lp}, align 8")
            return
        if isinstance(st, A.PtrRead):
            inn = ctx.ptr_inner.get(st.ptr)
            if not inn:
                raise LLVMGenError("ptrread needs a ptr local", st.loc)
            lp, _ = ctx.locals[st.ptr]
            pb = ctx.tmp()
            self.lines.append(f"  {pb} = load ptr, ptr {lp}, align 8")
            if inn.name in self.layouts:
                layout = self.layouts[inn.name]
                _, ft = layout.fields[0]
                inner_ty = f"%struct.{inn.name}"
                fg = ctx.tmp()
                self.lines.append(
                    f"  {fg} = getelementptr inbounds {inner_ty}, ptr {pb}, i32 0, i32 0"
                )
                lty = _scalar_ty(ft)
                v = self._load_typed_as_i32(fg, lty, ctx)
            else:
                lty = _scalar_ty(inn)
                v = self._load_typed_as_i32(pb, lty, ctx)
            dptr, dty = self._cursor_gep(ctx)
            self._store_i32_to_typed(dptr, dty, v, ctx)
            return
        if isinstance(st, A.PtrWrite):
            inn = ctx.ptr_inner.get(st.ptr)
            if not inn:
                raise LLVMGenError("*= needs a ptr local", st.loc)
            rhs, rty = self._emit_expr(st.value, ctx)
            lp, _ = ctx.locals[st.ptr]
            pb = ctx.tmp()
            self.lines.append(f"  {pb} = load ptr, ptr {lp}, align 8")
            if inn.name in self.layouts:
                layout = self.layouts[inn.name]
                inner_ty = f"%struct.{inn.name}"
                fg = ctx.tmp()
                self.lines.append(
                    f"  {fg} = getelementptr inbounds {inner_ty}, ptr {pb}, i32 0, i32 0"
                )
                _, ft = layout.fields[0]
                flty = _scalar_ty(ft)
                self._store_typed_at_ptr(fg, flty, rhs, rty, ctx)
            else:
                lty = _scalar_ty(inn)
                self._store_typed_at_ptr(pb, lty, rhs, rty, ctx)
            return
        if isinstance(st, A.FreeStmt):
            if st.ptr not in ctx.locals:
                raise LLVMGenError(f"free unknown {st.ptr}", st.loc)
            lp, _ = ctx.locals[st.ptr]
            p = ctx.tmp()
            self.lines.append(f"  {p} = load ptr, ptr {lp}, align 8")
            self.lines.append(f"  call void @free(ptr {p})")
            return
        raise LLVMGenError(type(st).__name__, getattr(st, "loc", None))

    def _emit_cond(self, c: A.Cond, v: str, ctx: _Ctx) -> str:
        t = ctx.tmp()
        if c.kind == ">0":
            self.lines.append(f"  {t} = icmp sgt i32 {v}, 0")
        elif c.kind == "<0":
            self.lines.append(f"  {t} = icmp slt i32 {v}, 0")
        elif c.kind == "==0":
            self.lines.append(f"  {t} = icmp eq i32 {v}, 0")
        elif c.kind == "!=0":
            self.lines.append(f"  {t} = icmp ne i32 {v}, 0")
        elif c.kind == ">N":
            self.lines.append(f"  {t} = icmp sgt i32 {v}, {int(c.imm)}")
        elif c.kind == "<N":
            self.lines.append(f"  {t} = icmp slt i32 {v}, {int(c.imm)}")
        elif c.kind == "==N":
            self.lines.append(f"  {t} = icmp eq i32 {v}, {int(c.imm)}")
        else:
            raise LLVMGenError(f"cond {c.kind}")
        return t

    def _gep(self, r: A.RefExpr, ctx: _Ctx) -> Tuple[str, str]:
        p = r.parts
        if not p or p[0] == "@":
            raise LLVMGenError("bad ref", r.loc)
        head = str(p[0])

        if head in ctx.ptr_struct:
            stn = ctx.ptr_struct[head]
            layout = self.layouts.get(stn)
            inner_ty = f"%struct.{stn}"
            if not layout or len(p) < 2 or not isinstance(p[1], str):
                raise LLVMGenError("ptr field ref", r.loc)
            fi = _struct_field_idx(layout, p[1])
            _, ft = layout.fields[fi]
            lp = ctx.locals[head][0]
            pb = ctx.tmp()
            self.lines.append(f"  {pb} = load ptr, ptr {lp}, align 8")
            fg = ctx.tmp()
            self.lines.append(
                f"  {fg} = getelementptr inbounds {inner_ty}, ptr {pb}, i32 0, i32 {fi}"
            )
            return (fg, _scalar_ty(ft))

        decl = ctx.local_segs.get(head)
        seg_ptr = ctx.local_seg_alloca.get(head) if decl else None
        if decl is None:
            decl = self.segs.get(head)
            if not decl:
                raise LLVMGenError(f"unknown segment {head}", r.loc)
            seg_sym = f"@{head}"
        else:
            seg_sym = seg_ptr or ""

        et = decl.elem_type
        arr_ty = self._seg_arr_ty(decl)
        stname = et.name
        inner_ty = f"%struct.{stname}" if stname in self.structs else _scalar_ty(et)
        layout = self.layouts.get(stname) if stname in self.layouts else None

        if len(p) == 1:
            g = ctx.tmp()
            self.lines.append(
                f"  {g} = getelementptr inbounds {arr_ty}, ptr {seg_sym}, i32 0, i32 0"
            )
            return (g, inner_ty)

        if len(p) >= 2 and isinstance(p[1], IntLit):
            idx = int(p[1].value)
            g = ctx.tmp()
            self.lines.append(
                f"  {g} = getelementptr inbounds {arr_ty}, ptr {seg_sym}, i32 0, i32 {idx}"
            )
            if len(p) == 2:
                return (g, inner_ty)
            if layout and len(p) == 3 and isinstance(p[2], str):
                fg = ctx.tmp()
                fi = _struct_field_idx(layout, p[2])
                _, ft = layout.fields[fi]
                self.lines.append(
                    f"  {fg} = getelementptr inbounds {inner_ty}, ptr {g}, i32 0, i32 {fi}"
                )
                return (fg, _scalar_ty(ft))
            raise LLVMGenError("ref shape", r.loc)

        if len(p) >= 2 and not isinstance(p[1], str):
            idxv, ity = self._emit_expr(p[1], ctx)
            iv = idxv
            if ity == "double":
                iv2 = ctx.tmp()
                self.lines.append(f"  {iv2} = fptosi double {idxv} to i32")
                iv = iv2
            elif ity != "i32":
                raise LLVMGenError("index must be i32", r.loc)
            g = ctx.tmp()
            self.lines.append(
                f"  {g} = getelementptr inbounds {arr_ty}, ptr {seg_sym}, i32 0, i32 {iv}"
            )
            if len(p) == 2:
                return (g, inner_ty)
            if layout and len(p) == 3 and isinstance(p[2], str):
                fg = ctx.tmp()
                fi = _struct_field_idx(layout, p[2])
                _, ft = layout.fields[fi]
                self.lines.append(
                    f"  {fg} = getelementptr inbounds {inner_ty}, ptr {g}, i32 0, i32 {fi}"
                )
                return (fg, _scalar_ty(ft))
            raise LLVMGenError("ref shape", r.loc)

        if layout and len(p) >= 2 and isinstance(p[1], str):
            g = ctx.tmp()
            self.lines.append(
                f"  {g} = getelementptr inbounds {arr_ty}, ptr {seg_sym}, i32 0, i32 0"
            )
            fi = _struct_field_idx(layout, p[1])
            _, ft = layout.fields[fi]
            fg = ctx.tmp()
            self.lines.append(
                f"  {fg} = getelementptr inbounds {inner_ty}, ptr {g}, i32 0, i32 {fi}"
            )
            return (fg, _scalar_ty(ft))

        raise LLVMGenError("unsupported ref for LLVM", r.loc)

    def _store_lhs(self, lhs: A.RefExpr, rhs: str, rty: str, ctx: _Ctx) -> None:
        ptr, lty = self._gep(lhs, ctx)
        al = _align(lty)
        if lty == "double" and rty == "i32":
            d = ctx.tmp()
            self.lines.append(f"  {d} = sitofp i32 {rhs} to double")
            self.lines.append(f"  store double {d}, ptr {ptr}, align {al}")
            sk = self._try_static_seg_slot(lhs, ctx)
            if sk:
                self._emit_maybe_watch(ctx, sk[0], sk[1])
            return
        if rty == "i32" and lty in ("i8", "i16"):
            t = ctx.tmp()
            op = "trunc" if lty == "i8" else "trunc"
            self.lines.append(f"  {t} = {op} i32 {rhs} to {lty}")
            self.lines.append(f"  store {lty} {t}, ptr {ptr}, align {al}")
            sk = self._try_static_seg_slot(lhs, ctx)
            if sk:
                self._emit_maybe_watch(ctx, sk[0], sk[1])
            return
        if lty == rty:
            self.lines.append(f"  store {lty} {rhs}, ptr {ptr}, align {al}")
            sk = self._try_static_seg_slot(lhs, ctx)
            if sk:
                self._emit_maybe_watch(ctx, sk[0], sk[1])
            return
        raise LLVMGenError(f"store {lty} <- {rty}")

    def _emit_expr(self, e: A.Expr, ctx: _Ctx) -> Tuple[str, str]:
        if isinstance(e, A.IntLit):
            return (str(int(e.value)), "i32")
        if isinstance(e, A.FloatLit):
            return (str(float(e.value)), "double")
        if isinstance(e, A.Ident):
            p, ty = ctx.locals[e.name]
            t = ctx.tmp()
            al = _align(ty)
            self.lines.append(f"  {t} = load {ty}, ptr {p}, align {al}")
            return (t, ty)
        if isinstance(e, A.BinOp):
            a, at = self._emit_expr(e.left, ctx)
            b, bt = self._emit_expr(e.right, ctx)
            t = ctx.tmp()
            if e.op in "+-*/" and at == "i32" and bt == "i32":
                op = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv"}[e.op]
                self.lines.append(f"  {t} = {op} nsw i32 {a}, {b}")
                return (t, "i32")
            if e.op in "+-*/":
                if at == "i32":
                    af = ctx.tmp()
                    self.lines.append(f"  {af} = sitofp i32 {a} to double")
                    a = af
                if bt == "i32":
                    bf = ctx.tmp()
                    self.lines.append(f"  {bf} = sitofp i32 {b} to double")
                    b = bf
                op = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}[e.op]
                self.lines.append(f"  {t} = {op} double {a}, {b}")
                return (t, "double")
        if isinstance(e, A.Unary) and e.op == "-":
            v, ty = self._emit_expr(e.expr, ctx)
            t = ctx.tmp()
            if ty == "i32":
                self.lines.append(f"  {t} = sub nsw i32 0, {v}")
            else:
                self.lines.append(f"  {t} = fneg double {v}")
            return (t, ty)
        if isinstance(e, A.Unary) and e.op == "&":
            if not isinstance(e.expr, A.RefExpr):
                raise LLVMGenError("& expects a ref (seg[i] or field path)", e.loc)
            addr, _ = self._gep(e.expr, ctx)
            return (addr, "ptr")
        if isinstance(e, A.Unary) and e.op == "*":
            if not isinstance(e.expr, A.Ident):
                raise LLVMGenError("* expects an identifier", e.loc)
            nm = e.expr.name
            inn = ctx.ptr_inner.get(nm)
            if not inn:
                raise LLVMGenError("* needs a ptr local", e.loc)
            lp, _ = ctx.locals[nm]
            pb = ctx.tmp()
            self.lines.append(f"  {pb} = load ptr, ptr {lp}, align 8")
            if inn.name in self.structs:
                layout = self.layouts[inn.name]
                _, ft = layout.fields[0]
                lty = _scalar_ty(ft)
                inner_ty = f"%struct.{inn.name}"
                fg = ctx.tmp()
                self.lines.append(
                    f"  {fg} = getelementptr inbounds {inner_ty}, ptr {pb}, i32 0, i32 0"
                )
                t = ctx.tmp()
                self.lines.append(f"  {t} = load {lty}, ptr {fg}, align {_align(lty)}")
                return (t, lty)
            lty = _scalar_ty(inn)
            t = ctx.tmp()
            self.lines.append(f"  {t} = load {lty}, ptr {pb}, align {_align(lty)}")
            return (t, lty)
        if isinstance(e, A.Call):
            return self._emit_call_expr(e, ctx, use_result=True)
        if isinstance(e, A.RefExpr):
            ptr, ty = self._gep(e, ctx)
            t = ctx.tmp()
            self.lines.append(f"  {t} = load {ty}, ptr {ptr}, align {_align(ty)}")
            return (t, ty)
        raise LLVMGenError(f"expr {type(e).__name__}", getattr(e, "loc", None))

    def _llvm_zext_i32_to_i64(self, v: str, ty: str, ctx: _Ctx) -> str:
        if ty == "i64":
            return v
        if ty == "i32":
            t = ctx.tmp()
            self.lines.append(f"  {t} = zext i32 {v} to i64")
            return t
        raise LLVMGenError(f"expected i32 or i64, got {ty}")

    def _emit_linux_libc_call(self, c: A.Call, ctx: _Ctx) -> Tuple[str, str]:
        n = c.name
        if n == "write" or n == "read":
            a0, t0 = self._emit_expr(c.args[0], ctx)
            a1, t1 = self._emit_expr(c.args[1], ctx)
            a2, t2 = self._emit_expr(c.args[2], ctx)
            a2r = self._llvm_zext_i32_to_i64(a2, t2, ctx)
            r = ctx.tmp()
            op = "write" if n == "write" else "read"
            self.lines.append(f"  {r} = call i64 @{op}(i32 {a0}, ptr {a1}, i64 {a2r})")
            return (r, "i64")
        if n == "open":
            parts: list[str] = []
            for arg in c.args:
                av, at = self._emit_expr(arg, ctx)
                if not parts:
                    parts.append(f"ptr {av}")
                else:
                    parts.append(f"i32 {av}")
            r = ctx.tmp()
            self.lines.append(f"  {r} = call i32 @open({', '.join(parts)})")
            return (r, "i32")
        if n == "close":
            a0, t0 = self._emit_expr(c.args[0], ctx)
            r = ctx.tmp()
            self.lines.append(f"  {r} = call i32 @close(i32 {a0})")
            return (r, "i32")
        if n == "exit":
            a0, t0 = self._emit_expr(c.args[0], ctx)
            self.lines.append(f"  call void @exit(i32 {a0})")
            return ("%z", "void")
        if n == "fork":
            r = ctx.tmp()
            self.lines.append(f"  {r} = call i32 @fork()")
            return (r, "i32")
        if n == "getpid":
            r = ctx.tmp()
            self.lines.append(f"  {r} = call i32 @getpid()")
            return (r, "i32")
        if n == "strlen":
            a0, t0 = self._emit_expr(c.args[0], ctx)
            r = ctx.tmp()
            self.lines.append(f"  {r} = call i64 @strlen(ptr {a0})")
            return (r, "i64")
        if n == "nanosleep":
            a0, t0 = self._emit_expr(c.args[0], ctx)
            a1, t1 = self._emit_expr(c.args[1], ctx)
            r = ctx.tmp()
            self.lines.append(f"  {r} = call i32 @nanosleep(ptr {a0}, ptr {a1})")
            return (r, "i32")
        raise LLVMGenError(f"unknown libc fn {n}", c.loc)

    def _emit_call_expr(self, c: A.Call, ctx: _Ctx, use_result: bool) -> Tuple[str, str]:
        if c.name == "sqrt" and len(c.args) == 1:
            v, ty = self._emit_expr(c.args[0], ctx)
            if ty == "i32":
                sv = ctx.tmp()
                self.lines.append(f"  {sv} = sitofp i32 {v} to double")
                v = sv
            t = ctx.tmp()
            self.lines.append(f"  {t} = call double @llvm.sqrt.f64(double {v})")
            return (t, "double")
        if getattr(self.mod, "use_linux_stdlib", False) and c.name in _LINUX_LIBC_NAMES:
            return self._emit_linux_libc_call(c, ctx)
        fd = self.fns.get(c.name)
        if not fd:
            raise LLVMGenError(f"unknown fn {c.name}", c.loc)
        parts: list[str] = []
        for i, (_, pt) in enumerate(fd.params):
            av, at = self._emit_expr(c.args[i], ctx)
            need = _scalar_ty(pt)
            if need == "double" and at == "i32":
                self.lines.append(f"  %ac{i} = sitofp i32 {av} to double")
                av = f"%ac{i}"
            parts.append(f"{need} {av}")
        rt = _scalar_ty(fd.ret)
        if rt == "void":
            self.lines.append(f"  call void @{c.name}({', '.join(parts)})")
            return ("%z", "void")
        t = ctx.tmp()
        self.lines.append(f"  {t} = call {rt} @{c.name}({', '.join(parts)})")
        return (t, rt)
