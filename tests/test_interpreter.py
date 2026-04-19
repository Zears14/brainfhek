import pytest

from bf2.interpreter import Interpreter, is_classic_bf, run_bf_classic
from bf2.parser import parse_source
from bf2.typechecker import check_module


def test_classic_bf_hello_piece():
    src = "++++++++++."
    assert is_classic_bf(src)
    out = run_bf_classic(src)
    assert len(out) == 1


def test_not_classic_with_keyword():
    assert not is_classic_bf("fn main() { }")


def test_reactor_no_recursion_on_nested_write():
    src = """
    seg a { i32[1] }
    seg b { i32[1] }
    watch a[0] {
        b[0] = 1
    }
    fn main() -> i32 {
        a[0] = 2
        ret 0
    }
    """
    mod = parse_source(src)
    check_module(mod)
    ip = Interpreter(mod)
    ip.run()
    assert ip.table.read_slot("b", 0) == 1


def test_struct_field_read():
    src = """
    struct P { x: i32, y: i32 }
    seg p { P }
    fn main() -> i32 {
        p.x = 3
        p.y = 4
        ret p.x + p.y
    }
    """
    mod = parse_source(src)
    check_module(mod)
    ip = Interpreter(mod)
    ip.run()


def test_pointer_and_field():
    src = r"""
    struct Point { x: i32, y: i32 }
    seg o { Point }
    fn main() -> i32 {
        o.x = 1
        o.y = 2
        ptr<Point> q = &o
        ret q.x + q.y
    }
    """
    mod = parse_source(src)
    check_module(mod)
    ip = Interpreter(mod)
    ip.run()


def test_main_returns():
    src = "fn main() -> i32 { ret 42 }"
    mod = parse_source(src)
    check_module(mod)
    ip = Interpreter(mod)
    assert ip.call_fn(next(x for x in mod.items if getattr(x, "name", None) == "main"), []) == 42
