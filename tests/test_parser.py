import pytest

from bf2.core.errors import BF2SyntaxError
from bf2.compiler.parser import parse_source


def test_parse_minimal_module():
    m = parse_source("struct S { x: i32 }\nseg g { i32[1] }\nfn main() -> i32 { ret 0 }\n")
    assert len(m.items) == 3


def test_syntax_error_not_exception():
    with pytest.raises(BF2SyntaxError) as ei:
        parse_source("fn x(")
    assert "BF2SyntaxError" in type(ei.value).__name__


def test_watch_parsed():
    m = parse_source("seg a { i32[1] }\nwatch a[0] { ret 0 }\nfn main() -> i32 { ret 0 }\n")
    assert len(m.items) == 3


def test_malformed_raises_bf2_syntax():
    with pytest.raises(BF2SyntaxError):
        parse_source("fn main() -> i32 { + * }")
