import pytest

from bf2.core.errors import BF2TypeError
from bf2.compiler.parser import parse_source
from bf2.compiler.typechecker import check_module


def test_typecheck_ok():
    src = """
    fn main() -> i32 { ret 0 }
    """
    check_module(parse_source(src))


def test_unknown_function_raises():
    src = """
    fn main() -> i32 { ret call nope(1) }
    """
    with pytest.raises(BF2TypeError):
        check_module(parse_source(src))


def test_segment_bounds_check():
    src = """
    seg s { i32[10] }
    fn main() -> i32 {
        s[10] = 5
        ret 0
    }
    """
    from bf2.compiler.diagnostics import DiagnosticCollector
    diag = DiagnosticCollector(enabled={"bounds"})
    check_module(parse_source(src), diag=diag)
    assert diag.has_warnings()
    assert any("out of bounds" in d.message for d in diag.diagnostics)
