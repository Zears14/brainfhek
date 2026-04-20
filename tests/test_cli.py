"""Tests for GCC/Clang-style CLI flag parsing."""

from __future__ import annotations

from bf2.cli import parse_compile_args


def test_parse_basic_path():
    opts = parse_compile_args(["hello.bf2"])
    assert opts.path == "hello.bf2"
    assert opts.opt_level == 0
    assert not opts.liro_enabled


def test_parse_output_flag():
    opts = parse_compile_args(["hello.bf2", "-o", "out.ll"])
    assert opts.path == "hello.bf2"
    assert opts.output == "out.ll"


def test_parse_optimization_levels():
    for level in (1, 2, 3):
        opts = parse_compile_args(["f.bf2", f"-O{level}"])
        assert opts.opt_level == level


def test_parse_fliro_default():
    opts = parse_compile_args(["f.bf2", "-fliro"])
    assert opts.liro_enabled
    assert opts.liro_spec == ""


def test_parse_fliro_spec():
    opts = parse_compile_args(["f.bf2", "-fliro=static_watch_fold,loop_metadata"])
    assert opts.liro_enabled
    assert opts.liro_spec == "static_watch_fold,loop_metadata"


def test_parse_fliro_index():
    opts = parse_compile_args(["f.bf2", "-fliro=0"])
    assert opts.liro_enabled
    assert opts.liro_spec == "0"


def test_parse_fno_liro():
    opts = parse_compile_args(["f.bf2", "-fliro", "-fno-liro"])
    assert not opts.liro_enabled


def test_parse_wall():
    opts = parse_compile_args(["f.bf2", "-Wall"])
    assert "ub" in opts.warnings
    assert "unreachable" in opts.warnings
    assert "shadow" in opts.warnings
    assert "unused" in opts.warnings


def test_parse_individual_warnings():
    opts = parse_compile_args(["f.bf2", "-Wub", "-Wunreachable"])
    assert opts.warnings == ["ub", "unreachable"]


def test_parse_wno_disables():
    opts = parse_compile_args(["f.bf2", "-Wall", "-Wno-unused"])
    assert "unused" not in opts.warnings
    assert "ub" in opts.warnings


def test_parse_target():
    opts = parse_compile_args(["f.bf2", "--target=aarch64-linux-gnu"])
    assert opts.target == "aarch64-linux-gnu"


def test_parse_additional_optflags():
    opts = parse_compile_args(["f.bf2", "--additional-optflags=-loop-unroll -simplifycfg"])
    assert opts.additional_optflags == "-loop-unroll -simplifycfg"


def test_parse_combined():
    opts = parse_compile_args([
        "examples/fib.bf2", "-O3", "-fliro=0", "-Wall", "-Wno-shadow",
        "-o", "out.ll", "--target=x86_64-linux-gnu",
    ])
    assert opts.path == "examples/fib.bf2"
    assert opts.opt_level == 3
    assert opts.liro_enabled
    assert opts.liro_spec == "0"
    assert "shadow" not in opts.warnings
    assert "ub" in opts.warnings
    assert opts.output == "out.ll"
    assert opts.target == "x86_64-linux-gnu"
