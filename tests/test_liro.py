"""Tests for the LIRO (LLVM IR Optimization) pass framework."""

from __future__ import annotations

import pytest
from llvmlite import binding

from bf2.liro import LIRO_REGISTRY, DEFAULT_LIRO_ORDER
from bf2.liro.runner import resolve_liro_spec, run_liros
from bf2.liro.static_watch_fold import StaticWatchFold
from bf2.liro.dead_branch_elim import DeadBranchElim
from bf2.liro.loop_metadata import LoopMetadata


def test_all_default_passes_registered():
    for name in DEFAULT_LIRO_ORDER:
        assert name in LIRO_REGISTRY


def test_registry_classes():
    assert LIRO_REGISTRY["static_watch_fold"] is StaticWatchFold
    assert LIRO_REGISTRY["dead_branch_elim"] is DeadBranchElim
    assert LIRO_REGISTRY["loop_metadata"] is LoopMetadata


def test_resolve_empty_returns_defaults():
    assert resolve_liro_spec("") == DEFAULT_LIRO_ORDER


def test_resolve_index():
    assert resolve_liro_spec("0") == [DEFAULT_LIRO_ORDER[0]]
    assert resolve_liro_spec("2") == [DEFAULT_LIRO_ORDER[2]]


def test_resolve_index_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_liro_spec("99")


def test_resolve_named():
    result = resolve_liro_spec("loop_metadata,static_watch_fold")
    assert result == ["loop_metadata", "static_watch_fold"]


def test_resolve_unknown_name():
    with pytest.raises(ValueError, match="Unknown LIRO"):
        resolve_liro_spec("nonexistent_pass")


def test_run_liros_with_valid_ir():
    ir = """
; ModuleID = 'test'
target triple = "x86_64-pc-linux-gnu"
define i32 @main() {
entry:
  ret i32 0
}
"""
    result = run_liros(ir, [])
    assert "ret i32 0" in result


def test_loop_metadata_with_comments():
    """Verify that LoopMetadata handles labels and branches with trailing comments."""
    ir = """define void @test() {
entry:
  br label %loop_start ; jump to loop
loop_start:             ; preds = %body, %entry
  %c = icmp eq i32 0, 0
  br i1 %c, label %body, label %end
body:                   ; preds = %loop_start
  br label %loop_start  ; back-edge
end:                    ; preds = %loop_start
  ret void
}"""
    result = LoopMetadata().run(binding.parse_assembly(ir))
    joined = str(result)
    assert "!llvm.loop" in joined
    assert "loop_start" in joined
    # Check that the back-edge branch was annotated
    assert "br label %loop_start, !llvm.loop !" in joined
