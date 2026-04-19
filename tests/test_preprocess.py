from pathlib import Path

import pytest

from bf2.parser import parse_source
from bf2.preprocess import PreprocessError, preprocess, preprocess_path
from bf2.typechecker import check_module


def test_include_stdlib_sets_flag_and_defines_expand(tmp_path):
    main = tmp_path / "user.bf2"
    src = """
#include "stdlib"
fn main() -> i32 {
  ret STDOUT
}
"""
    main.write_text(src, encoding="utf-8")
    text, meta = preprocess(main.read_text(encoding="utf-8"), main_path=main.resolve())
    assert meta.use_linux_stdlib is True
    assert "STDOUT" not in text
    assert "ret 1" in text
    mod = parse_source(text, use_linux_stdlib=meta.use_linux_stdlib)
    check_module(mod)


def test_preprocess_path_resolves_bundled_stdlib(tmp_path):
    stub = tmp_path / "t.bf2"
    stub.write_text('#include "stdlib"\nfn main() -> i32 { ret 0 }\n', encoding="utf-8")
    _, meta = preprocess_path(stub.resolve())
    assert meta.use_linux_stdlib is True


def test_include_missing_raises():
    with pytest.raises(PreprocessError):
        preprocess('#include "does_not_exist_99.bf2"\n', main_path=Path("/tmp/x.bf2"))
