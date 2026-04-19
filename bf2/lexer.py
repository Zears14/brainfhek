from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

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
    value: object
    line: int
    col: int


class Lexer:
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
            if self.text.startswith("//", self.i):
                while self.i < len(self.text) and self._peek() != "\n":
                    self._adv()
                continue
            if self.text.startswith("/*", self.i):
                self._adv(2)
                while self.i < len(self.text) and not self.text.startswith("*/", self.i):
                    self._adv()
                if self.text.startswith("*/", self.i):
                    self._adv(2)
                continue
            if c in "><+-.,[]{}()@=;":
                sym = c
                self._adv()
                yield Token("SYMBOL", sym, start_line, start_col)
                continue
            if c in "*/":
                self._adv()
                yield Token("SYMBOL", c, start_line, start_col)
                continue
            if c == "!":
                if self._peek(1) == "=":
                    self._adv(2)
                    yield Token("SYMBOL", "!=", start_line, start_col)
                else:
                    self._adv()
                    yield Token("SYMBOL", "!", start_line, start_col)
                continue
            if c in "<>":
                nxt = self._peek(1)
                if nxt == "=":
                    self._adv(2)
                    yield Token("SYMBOL", c + "=", start_line, start_col)
                elif c == "<" and nxt == "<":
                    self._adv(2)
                    yield Token("SYMBOL", "<<", start_line, start_col)
                elif c == ">" and nxt == ">":
                    self._adv(2)
                    yield Token("SYMBOL", ">>", start_line, start_col)
                else:
                    self._adv()
                    yield Token("SYMBOL", c, start_line, start_col)
                continue
            if c == "=":
                if self._peek(1) == "=":
                    self._adv(2)
                    yield Token("SYMBOL", "==", start_line, start_col)
                else:
                    self._adv()
                    yield Token("SYMBOL", "=", start_line, start_col)
                continue
            if c == ":":
                self._adv()
                yield Token("SYMBOL", ":", start_line, start_col)
                continue
            if c == ",":
                self._adv()
                yield Token("SYMBOL", ",", start_line, start_col)
                continue
            if c in "0123456789":
                j = self.i
                while self._peek().isdigit():
                    self._adv()
                num_s = self.text[j : self.i]
                if self._peek() == "." and self._peek(1).isdigit():
                    self._adv()
                    while self._peek().isdigit():
                        self._adv()
                    yield Token("FLOAT", float(self.text[j : self.i]), start_line, start_col)
                else:
                    yield Token("INT", int(num_s), start_line, start_col)
                continue
            if c.isalpha() or c == "_":
                j = self.i
                while self._peek().isalnum() or self._peek() == "_":
                    self._adv()
                name = self.text[j : self.i]
                if name in KEYWORDS:
                    yield Token("KEYWORD", name, start_line, start_col)
                elif name in TYPES:
                    yield Token("TYPE", name, start_line, start_col)
                else:
                    yield Token("IDENT", name, start_line, start_col)
                continue
            if c == '"':
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
                yield Token("STRING", "".join(buf), start_line, start_col)
                continue
            self._adv()
            yield Token("SYMBOL", c, start_line, start_col)
        yield Token("EOF", None, self.line, self.col)
