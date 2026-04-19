import pytest

from bf2.lexer import Lexer, Token


def test_token_kinds_basic():
    toks = list(Lexer("fn main() { }").tokenize())
    kinds = [t.kind for t in toks if t.value != "\n"]
    assert "KEYWORD" in kinds and "IDENT" in kinds


def test_newline_token_value():
    toks = list(Lexer("a\nb").tokenize())
    assert any(t.kind == "SYMBOL" and t.value == "\n" for t in toks)


def test_comment_line_skipped():
    toks = list(Lexer("// x\nfn").tokenize())
    assert any(t.kind == "KEYWORD" and t.value == "fn" for t in toks)


def test_block_comment_skipped():
    toks = list(Lexer("/* c */ seg").tokenize())
    assert any(t.kind == "KEYWORD" and t.value == "seg" for t in toks)


def test_float_token():
    toks = list(Lexer("3.14").tokenize())
    assert toks[0].kind == "FLOAT"


def test_string_literal():
    toks = list(Lexer('"hi"').tokenize())
    assert toks[0].kind == "STRING" and toks[0].value == "hi"
