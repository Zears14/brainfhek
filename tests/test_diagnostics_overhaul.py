import pytest
from bf2.compiler.parser import parse_source
from bf2.compiler.typechecker import check_module
from bf2.compiler.diagnostics import DiagnosticCollector

def check_src(src):
    mod = parse_source(src)
    dc = DiagnosticCollector(enabled={"ub", "bounds", "types"})
    check_module(mod, diag=dc)
    return dc.diagnostics

def test_div_by_zero():
    src = "fn main() -> i32 { i32 x = 10 / 0 ret 0 }"
    ds = check_src(src)
    assert any("division by zero" in d.message for d in ds)

def test_bitwise_float():
    src = "fn main() -> i32 { f32 x = 1.0 & 2.0 ret 0 }"
    ds = check_src(src)
    assert any("bitwise operation '&' on floating-point" in d.message for d in ds)

def test_shift_ub():
    src = "fn main() -> i32 { i32 x = 1 << -1 ret 0 }"
    ds = check_src(src)
    assert any("negative shift count" in d.message for d in ds)
    
    src = "fn main() -> i32 { i32 x = 1 << 64 ret 0 }"
    ds = check_src(src)
    assert any("shift count 64 is too large" in d.message for d in ds)

def test_call_args_mismatch():
    src = """
    fn foo(a: i32, b: i32) -> i32 { ret a + b }
    fn main() -> i32 { 
        call foo(1) 
        ret 0 
    }
    """
    ds = check_src(src)
    assert any("expects 2 arguments but got 1" in d.message for d in ds)

def test_assignment_mismatch():
    src = """
    struct A { x: i32 }
    fn main() -> i32 {
        i32 x = 10
        ptr<A> p = 0
        x = p // Warning
        ret 0
    }
    """
    ds = check_src(src)
    assert any("assignment between potentially incompatible types" in d.message for d in ds)

def test_static_oob():
    src = """
    seg a { i32[10] }
    fn main() -> i32 {
        a[10] = 1 // Warning
        ret 0
    }
    """
    ds = check_src(src)
    assert any("index 10 is out of bounds" in d.message for d in ds)
