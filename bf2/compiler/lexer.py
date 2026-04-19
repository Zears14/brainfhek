from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

KEYWORDS = frozenset(
    {
        "fn",
        "ret",
        "call",
        "seg",
        "struct",
        "watch",
        "load",
        "store",
        "swap",
        "label",
        "jump",
        "if",
        "else",
        "alloc",
        "as",
        "free",
        "type",
        "ptr",
        "ptrread",
        "do",
        "true",
        "false",
    }
)

TYPES = frozenset({"i8", "i16", "i32", "i64", "f32", "f64", "bool"})


@dataclass
class Token:
    kind: str
    value: Optional[object]
    line: int
    col: int


class Lexer:
    """Tokenizes Brainfhek source code."""

    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.line = 1
        self.col = 1

    def _adv(self, n: int = 1) -> None:
        for _ in range(n):
            if self.i < len(self.text) and self.text[self.i] == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.i += 1

    def _peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.text[j] if j < len(self.text) else ""

    def _match_string(self, s: str) -> bool:
        return self.text.startswith(s, self.i)

    def tokenize(self) -> Iterator[Token]:
        while self.i < len(self.text):
            c = self._peek()
            start_line, start_col = self.line, self.col

            if c in " \t\r":
                self._adv()
                continue
            if c == "\n":
                yield Token("SYMBOL", "\n", start_line, start_col)
                self._adv()
                continue
            if self._match_string("#include"):
                self._adv(8)
                yield Token("KEYWORD", "#include", start_line, start_col)
                continue
            if self._match_string("//"):
                self._skip_comment()
                continue
            if self._match_string("/*"):
                self._skip_block_comment()
                continue
            if c in "><+-.,[]{}()@=;":
                yield from self._read_operator(c, start_line, start_col)
                continue
            if c in "*/":
                self._adv()
                yield Token("SYMBOL", c, start_line, start_col)
                continue
            if c == "!":
                yield from self._read_bang(start_line, start_col)
                continue
            if c == ":":
                self._adv()
                yield Token("SYMBOL", ":", start_line, start_col)
                continue
            if c in "0123456789":
                yield self._read_number(start_line, start_col)
                continue
            if c.isalpha() or c == "_":
                yield self._read_name_or_keyword(start_line, start_col)
                continue
            if c == '"':
                yield self._read_string_literal(start_line, start_col)
                continue

            # Fallback for unknown symbols
            self._adv()
            yield Token("SYMBOL", c, start_line, start_col)

        yield Token("EOF", None, self.line, self.col)

    def _skip_comment(self) -> None:
        while self.i < len(self.text) and self._peek() != "\n":
            self._adv()

    def _skip_block_comment(self) -> None:
        self._adv(2)
        while self.i < len(self.text) and not self._match_string("*/"):
            self._adv()
        if self._match_string("*/"):
            self._adv(2)

    def _read_operator(self, c: str, line: int, col: int) -> Iterator[Token]:
        if c in "<>":
            self._peek(1)
            # In BF2, we don't want to combine << or >> because they are used in nested pointers ptr<ptr<T>>
            # and move-rel operations like >> (move 2 right).
            # However, >= and <= ARE combined in some languages. BF2 uses >0, ==0 etc.
            # SPEC doesn't show >= or <= operators.
            self._adv()
            yield Token("SYMBOL", c, line, col)
        elif c == "=":
            if self._peek(1) == "=":
                self._adv(2)
                yield Token("SYMBOL", "==", line, col)
            else:
                self._adv()
                yield Token("SYMBOL", "=", line, col)
        elif c == "-":
            if self._peek(1) == ">":
                self._adv(2)
                yield Token("SYMBOL", "->", line, col)
            else:
                self._adv()
                yield Token("SYMBOL", "-", line, col)
        else:
            self._adv()
            yield Token("SYMBOL", c, line, col)

    def _read_bang(self, line: int, col: int) -> Iterator[Token]:
        if self._peek(1) == "=":
            self._adv(2)
            yield Token("SYMBOL", "!=", line, col)
        else:
            self._adv()
            yield Token("SYMBOL", "!", line, col)

    def _read_number(self, line: int, col: int) -> Token:
        start = self.i
        while self._peek().isdigit():
            self._adv()
        num_s = self.text[start : self.i]
        
        if self._peek() == "." and self._peek(1).isdigit():
            self._adv()
            while self._peek().isdigit():
                self._adv()
            return Token("FLOAT", float(self.text[start : self.i]), line, col)
        else:
            return Token("INT", int(num_s), line, col)

    def _read_name_or_keyword(self, line: int, col: int) -> Token:
        start = self.i
        while self._peek().isalnum() or self._peek() == "_":
            self._adv()
        name = self.text[start : self.i]
        
        if name in KEYWORDS:
            return Token("KEYWORD", name, line, col)
        elif name in TYPES:
            return Token("TYPE", name, line, col)
        else:
            return Token("IDENT", name, line, col)

    def _read_string_literal(self, line: int, col: int) -> Token:
        self._adv()
        buf: list[str] = []
        while self.i < len(self.text) and self._peek() != '"':
            if self._peek() == "\\":
                self._adv()
                esc = self._peek()
                self._adv()
                buf.append({"n": "\n", "t": "\t", "r": "\r", '"': '"'}.get(esc, esc))
            else:
                buf.append(self._peek())
                self._adv()
        if self._peek() == '"':
            self._adv()
        return Token("STRING", "".join(buf), line, col)
