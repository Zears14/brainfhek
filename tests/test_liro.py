"""Tests for the LIRO (LLVM IR Optimization) pass framework."""

from __future__ import annotations

import pytest

from bf2.liro import LIRO_REGISTRY, DEFAULT_LIRO_ORDER
from bf2.liro.runner import resolve_liro_spec, run_liros
from bf2.liro.static_watch_fold import StaticWatchFold
from bf2.liro.dead_branch_elim import DeadBranchElim
from bf2.liro.loop_metadata import LoopMetadata


# --- Registry tests ---


def test_all_default_passes_registered():
    for name in DEFAULT_LIRO_ORDER:
        assert name in LIRO_REGISTRY


def test_registry_classes():
    assert LIRO_REGISTRY["static_watch_fold"] is StaticWatchFold
    assert LIRO_REGISTRY["dead_branch_elim"] is DeadBranchElim
    assert LIRO_REGISTRY["loop_metadata"] is LoopMetadata


# --- Spec resolution tests ---


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


# --- StaticWatchFold tests ---


def test_static_watch_fold_constant_true():
    lines = [
        "  %t1 = icmp eq i32 0, 0",
        "  br i1 %t1, label %fire, label %skip",
    ]
    result = StaticWatchFold().run(lines)
    assert "  br label %fire" in result
    assert "icmp eq" not in "\n".join(result)


def test_static_watch_fold_constant_false():
    lines = [
        "  %t1 = icmp eq i32 5, 0",
        "  br i1 %t1, label %fire, label %skip",
    ]
    result = StaticWatchFold().run(lines)
    assert "  br label %skip" in result


def test_static_watch_fold_and_propagation():
    lines = [
        "  %t1 = icmp eq i32 5, 0",
        "  %t2 = load i32, ptr @bf2.watch.depth, align 4",
        "  %t3 = icmp slt i32 %t2, 8",
        "  %t4 = and i1 %t1, %t3",
        "  br i1 %t4, label %fire, label %skip",
    ]
    result = StaticWatchFold().run(lines)
    # t1 is false, so t4 = false AND x = false → branch to skip
    assert "  br label %skip" in result


def test_static_watch_fold_no_constants():
    lines = [
        "  %t1 = icmp eq i32 %a, %b",
        "  br i1 %t1, label %fire, label %skip",
    ]
    result = StaticWatchFold().run(lines)
    assert result == lines  # unchanged


# --- DeadBranchElim tests ---


def test_dead_branch_removes_unreachable():
    lines = [
        "define void @test() {",
        "entry:",
        "  br label %live",
        "dead_block:",
        "  call void @unreachable_fn()",
        "  br label %live",
        "live:",
        "  ret void",
        "}",
    ]
    result = DeadBranchElim().run(lines)
    joined = "\n".join(result)
    assert "dead_block:" not in joined
    assert "@unreachable_fn" not in joined
    assert "live:" in joined
    assert "ret void" in joined


def test_dead_branch_keeps_all_reachable():
    lines = [
        "define void @test() {",
        "entry:",
        "  br i1 %c, label %a, label %b",
        "a:",
        "  br label %end",
        "b:",
        "  br label %end",
        "end:",
        "  ret void",
        "}",
    ]
    result = DeadBranchElim().run(lines)
    assert result == lines  # nothing removed


# --- LoopMetadata tests ---


def test_loop_metadata_annotates_backedge():
    lines = [
        "define void @test() {",
        "entry:",
        "  br label %loop.head",
        "loop.head:",
        "  br i1 %c, label %loop.body, label %loop.end",
        "loop.body:",
        "  br label %loop.head",
        "loop.end:",
        "  ret void",
        "}",
    ]
    result = LoopMetadata().run(lines)
    joined = "\n".join(result)
    assert "!llvm.loop" in joined
    assert "llvm.loop.vectorize.enable" in joined
    assert "llvm.loop.unroll.enable" in joined


def test_loop_metadata_no_loops():
    lines = [
        "define void @test() {",
        "entry:",
        "  ret void",
        "}",
    ]
    result = LoopMetadata().run(lines)
    assert result == lines  # unchanged


# --- Integration: run_liros ---


def test_run_liros_pipeline():
    """Watch fold + dead branch elim should clean up a static-false watch."""
    ir = """define void @main() {
entry:
  %t1 = icmp eq i32 5, 0
  br i1 %t1, label %fire, label %skip
fire:
  call void @bf2.watch.0()
  br label %join
skip:
  br label %join
join:
  ret void
}"""
    result = run_liros(ir, ["static_watch_fold", "dead_branch_elim"])
    assert "call void @bf2.watch.0()" not in result
    assert "fire:" not in result
    assert "ret void" in result
