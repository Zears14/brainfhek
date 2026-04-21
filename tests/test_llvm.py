import shutil
import subprocess
from pathlib import Path

import pytest

from bf2.compiler.parser import parse_source
from bf2.compiler.typechecker import check_module
from bf2.compiler.preprocess import preprocess
from bf2.backends.llvm.emitter import emit_llvm_ir


def _emit(text: str, main_path: Path | None = None) -> str:
    text, meta = preprocess(text, main_path=main_path)
    mod = parse_source(text, use_linux_stdlib=meta.use_linux_stdlib)
    check_module(mod)
    return str(emit_llvm_ir(mod))


def _compile_and_run(example: Path, *args: str) -> subprocess.CompletedProcess[str]:
    ir = _emit(example.read_text(encoding="utf-8"), main_path=example)
    stem = example.stem
    ll = Path(f"/tmp/bf2_{stem}.ll")
    ll.write_text(ir, encoding="utf-8")
    out = Path(f"/tmp/bf2_{stem}.out")
    subprocess.run(
        ["clang", "-O1", str(ll), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    return subprocess.run([str(out), *args], capture_output=True, text=True)


def test_emit_fibonacci_compiles():
    p = Path(__file__).resolve().parents[1] / "examples" / "fibonacci.bf2"
    ir = _emit(p.read_text(encoding="utf-8"), main_path=p)
    assert '@"fib"' in ir


def test_all_examples_compile():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    for example in sorted(examples_dir.glob("*.bf2")):
        ir = _emit(example.read_text(encoding="utf-8"), main_path=example)
        assert "define" in ir


def test_emit_watch_and_globals():
    src = """
seg vals { i64[1] }
watch vals[0] { if(> 0) { vals[0] = 1 } }
fn main() -> i32 { vals[0] = 0 ret 0 }
"""
    ir = _emit(src)
    assert '@"vals"' in ir
    assert '@"bf2.watch.0"' in ir
    assert '@"bf2.watch.depth"' in ir


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not in PATH")
def test_clang_roundtrip_fibonacci():
    p = Path(__file__).resolve().parents[1] / "examples" / "fibonacci-small.bf2"
    r = _compile_and_run(p)
    assert r.returncode == 0
    lines = [x for x in r.stdout.strip().split("\n") if x.strip()]
    assert lines[0].strip() == "0: 0"
    assert lines[1].strip() == "1: 1"
    assert lines[9].strip() == "9: 34"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not in PATH")
def test_clang_roundtrip_brainfuck_hello():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    r = _compile_and_run(examples_dir / "brainfuck.bf2", str(examples_dir / "hello.bf"))
    assert r.returncode == 0
    assert r.stdout == "Hello World!\n"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not in PATH")
def test_clang_roundtrip_reactive_clamp():
    p = Path(__file__).resolve().parents[1] / "examples" / "reactive_clamp.bf2"
    r = _compile_and_run(p)
    assert r.returncode == 0
    assert r.stdout == "50\n100\n0\n"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not in PATH")
def test_clang_roundtrip_struct_point():
    p = Path(__file__).resolve().parents[1] / "examples" / "struct_point.bf2"
    r = _compile_and_run(p)
    assert r.returncode == 0
    assert r.stdout == "5\n"


def test_struct_point_distance():
    p = Path(__file__).resolve().parents[1] / "examples" / "struct_point.bf2"
    ir = _emit(p.read_text(encoding="utf-8"), main_path=p)
    assert '@"distance"' in ir


def test_emit_alloc_exprstmt_dynamic_move_getchar():
    src = """
seg buf { i8[1] }
fn main() -> i32 {
  buf[0] = 65
  ret 0
}
"""
    ir = _emit(src)
    assert "buf" in ir
    assert "main" in ir


def test_linux_stdlib_ir(tmp_path):
    main = tmp_path / "with_stdlib.bf2"
    main.write_text(
        '#include "stdlib"\nfn main() -> i32 {\n  ret 0\n}\n',
        encoding="utf-8",
    )
    text, meta = preprocess(main.read_text(encoding="utf-8"), main_path=main.resolve())
    mod = parse_source(text, use_linux_stdlib=meta.use_linux_stdlib)
    check_module(mod)
    ir = str(emit_llvm_ir(mod))
    assert "snprintf" in ir
    assert "write" in ir
    assert "nanosleep" in ir
    assert "__bf2_fmt" in ir
    assert "STDOUT" in ir
    assert "O_CREAT" in ir
