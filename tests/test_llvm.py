"""LLVM IR emit tests; optional clang link if available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bf2.backends.llvm.emitter import emit_llvm_ir
from bf2.compiler.parser import parse_source
from bf2.compiler.preprocess import preprocess
from bf2.compiler.typechecker import check_module


def _emit(src: str, *, main_path: Path | None = None) -> str:
    if main_path is not None:
        text, meta = preprocess(src, main_path=main_path.resolve())
        mod = parse_source(text, use_linux_stdlib=meta.use_linux_stdlib)
    else:
        mod = parse_source(src)
    check_module(mod)
    return emit_llvm_ir(mod)


def test_emit_fibonacci_compiles():
    src = """
fn fib(n: i32) -> i32 {
    if(== 0) { ret 0 }
    if(== 1) { ret 1 }
    ret call fib(n - 1) + call fib(n - 2)
}
fn main() -> i32 { ret 0 }
"""
    ir = _emit(src)
    assert "define i32 @fib(" in ir
    assert "call i32 @fib(" in ir


def test_emit_watch_and_globals():
    src = """
seg vals { i32[1] }
watch vals[0] { if(> 0) { vals[0] = 1 } }
fn main() -> i32 { vals[0] = 0 ret 0 }
"""
    ir = _emit(src)
    assert "@vals = global" in ir
    assert "define void @bf2.watch.0()" in ir
    assert "@bf2.watch.depth = global" in ir


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not in PATH")
def test_clang_roundtrip_fibonacci():
    p = Path(__file__).resolve().parents[1] / "examples" / "fibonacci.bf2"
    ir = _emit(p.read_text(encoding="utf-8"), main_path=p)
    ll = Path("/tmp/bf2_test_fib.ll")
    ll.write_text(ir, encoding="utf-8")
    out = Path("/tmp/bf2_test_fib.out")
    subprocess.run(
        ["clang", "-O1", str(ll), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    r = subprocess.run([str(out)], capture_output=True, text=True)
    assert r.returncode == 0
    lines = [x for x in r.stdout.strip().split("\n") if x.strip()]
    assert lines[0].strip().endswith("0")
    assert lines[1].strip().endswith("1")
    assert lines[9].strip().endswith("34")


def test_struct_point_distance():
    p = Path(__file__).resolve().parents[1] / "examples" / "struct_point.bf2"
    ir = _emit(p.read_text(encoding="utf-8"), main_path=p)
    assert "define double @distance(" in ir
    assert "llvm.sqrt.f64" in ir


def test_emit_alloc_exprstmt_dynamic_move_getchar():
    src = """
seg buf { i32[8] }
fn main() -> i32 {
  do call sqrt(4.0)
  alloc i32 4 as p
  @buf[1+0]
  , 
  ret 0
}
"""
    ir = _emit(src)
    assert "call ptr @malloc" in ir
    assert "call double @llvm.sqrt.f64" in ir
    assert "call i32 @getchar()" in ir
    assert "add nsw i32" in ir


def test_linux_stdlib_ir(tmp_path):
    main = tmp_path / "with_stdlib.bf2"
    main.write_text(
        '#include "stdlib"\nfn main() -> i32 {\n  ret 0\n}\n',
        encoding="utf-8",
    )
    text, meta = preprocess(main.read_text(encoding="utf-8"), main_path=main.resolve())
    mod = parse_source(text, use_linux_stdlib=meta.use_linux_stdlib)
    check_module(mod)
    ir = emit_llvm_ir(mod)
    assert "declare i32 @snprintf(" in ir
    assert "declare i64 @write(" in ir
    assert "declare i32 @nanosleep(" in ir
    assert "declare i32 @printf(" not in ir
    assert "@STDOUT = global i32 1" in ir
    assert "@O_CREAT = global i32 64" in ir
    assert "%struct.timespec = type { i64, i64 }" in ir
