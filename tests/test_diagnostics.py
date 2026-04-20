"""Tests for the diagnostic collector and typechecker warning emission."""

from __future__ import annotations

from bf2.compiler.diagnostics import DiagnosticCollector
from bf2.compiler.parser import parse_source
from bf2.compiler.typechecker import check_module
from bf2.core.errors import SourceLoc


# --- DiagnosticCollector unit tests ---


def test_warn_enabled():
    dc = DiagnosticCollector(enabled={"ub"})
    dc.warn("ub", "division by zero", SourceLoc(1, 1))
    assert len(dc.diagnostics) == 1
    assert dc.diagnostics[0].code == "ub"


def test_warn_disabled():
    dc = DiagnosticCollector(enabled=set())
    dc.warn("ub", "division by zero", SourceLoc(1, 1))
    assert len(dc.diagnostics) == 0


def test_note_always_recorded():
    dc = DiagnosticCollector(enabled=set())
    dc.note("some note")
    assert len(dc.diagnostics) == 1


def test_has_warnings():
    dc = DiagnosticCollector(enabled={"ub"})
    assert not dc.has_warnings()
    dc.warn("ub", "test")
    assert dc.has_warnings()


def test_format_all():
    dc = DiagnosticCollector(enabled={"ub"})
    dc.warn("ub", "division by zero", SourceLoc(1, 5))
    result = dc.format_all("x = y / 0")
    assert "warning[-Wub]" in result
    assert "division by zero" in result
    assert "x = y / 0" in result


# --- Typechecker warning integration tests ---


def _check_with_warnings(src: str, warnings: set[str]) -> DiagnosticCollector:
    mod = parse_source(src)
    dc = DiagnosticCollector(enabled=warnings)
    check_module(mod, diag=dc)
    return dc


def test_warn_unreachable_code():
    src = """
fn main() -> i32 {
    ret 0
    ret 1
}
"""
    dc = _check_with_warnings(src, {"unreachable"})
    msgs = [d.message for d in dc.diagnostics]
    assert any("unreachable" in m for m in msgs)


def test_warn_division_by_zero():
    src = """
fn main() -> i32 {
    ret 10 / 0
}
"""
    dc = _check_with_warnings(src, {"ub"})
    msgs = [d.message for d in dc.diagnostics]
    assert any("division by zero" in m for m in msgs)


def test_no_warnings_when_disabled():
    src = """
fn main() -> i32 {
    ret 0
    ret 1
}
"""
    dc = _check_with_warnings(src, set())
    assert len(dc.diagnostics) == 0
