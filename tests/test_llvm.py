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


def test_sext_for_signed_coercion_in_binop():
    src = """
fn main() -> i32 {
    seg buf { i64[2] }
    i32 neg = -5
    i32 pos = 10
    buf[0] = neg + pos
    i64 result = buf[0]
    ret 0
}
"""
    ir = _emit(src)
    assert "sext i32" in ir
    assert "zext i32" not in ir


def test_sext_for_signed_coercion_in_assign():
    src = """
fn main() -> i32 {
    i32 x = -100
    i64 y = x
    ret 0
}
"""
    ir = _emit(src)
    assert "sext i32" in ir
    assert "zext i32" not in ir


def test_sext_for_i32_to_i64_call_args():
    src = """
    #include "stdlib"
    fn main() -> i32 {
        i32 n = -1
        call write(1, "test", n)
        ret 0
    }
    """
    ir = _emit(src)
    assert "sext i32" in ir
    assert "zext i32" not in ir


def test_no_orphan_cslot_global():
    src = "fn main() -> i32 { ret 0 }"
    ir = _emit(src)
    assert '@"__cslot"' not in ir


def test_local_seg_initialized():
    src = """
    fn main() -> i32 {
        seg buf { i32[16] }
        seg result { i64[1] }
        result[0] = buf[0] + buf[1]
        ret 0
    }
    """
    ir = _emit(src)
    assert "zeroinitializer" in ir


def test_nounwind_on_all_functions():
    src = """
    fn helper(n: i32) -> i32 { ret n }
    fn main() -> i32 { ret 0 }
    """
    ir = _emit(src)
    assert "nounwind" in ir
    assert ir.count("nounwind") >= 2  # Both helper and main
    assert "#0" not in ir  # No attribute groups


def test_type_cache_isolation():
    src1 = "fn main() -> i32 { ret 0 }"
    src2 = "fn main() -> i64 { ret 0 }"
    ir1 = _emit(src1)
    ir2 = _emit(src2)
    # Both should compile without issues, types should be correct
    assert 'define i32 @"main"() nounwind' in ir1
    assert 'define i64 @"main"() nounwind' in ir2


def test_alloca_declaration_order():
    src = """
    fn f(n: i32) -> i32 {
        seg a { i32[2] }
        seg b { i32[2] }
        ret 0
    }
    fn main() -> i32 { ret 0 }
    """
    ir = _emit(src)
    # Find positions of declarations
    cseg_pos = ir.index('"__cseg"')
    cslot_pos = ir.index('"__cslot"')
    p_n_pos = ir.index('.p.n"')
    lseg_a_pos = ir.index('.lseg.a"')
    lseg_b_pos = ir.index('.lseg.b"')
    # Verify forward order: cseg < cslot < p.n < lseg.a < lseg.b
    assert cseg_pos < cslot_pos < p_n_pos < lseg_a_pos < lseg_b_pos


def test_loop_metadata():
    src_bf = "fn main() -> i32 { [ { } ] ret 0 }"
    src_counted = "fn main() -> i32 { { 5 } { } ret 0 }"
    ir_bf = _emit(src_bf)
    ir_counted = _emit(src_counted)
    assert "!llvm.loop" in ir_bf
    assert "!llvm.loop" in ir_counted


def test_loop_counted_i64():
    src = "fn main() -> i32 { { 3000000000 } { } ret 0 }"
    ir = _emit(src)
    assert "icmp slt i64" in ir
