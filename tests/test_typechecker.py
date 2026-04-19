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
