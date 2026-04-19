from __future__ import annotations

from typing import List, Optional, Union

from bf2 import ast_nodes as A
from bf2.errors import BF2SyntaxError, SourceLoc
from bf2.lexer import Lexer, Token


def _loc(t: Token) -> SourceLoc:
    return SourceLoc(t.line, t.col)


class Parser:
    def __init__(self, tokens: List[Token], text: str):
        self.toks = tokens
        self.pos = 0
        self.text = text

    def _skip_nl(self) -> None:
        while self._cur().kind == "SYMBOL" and self._cur().value == "\n":
            self.pos += 1

    def _peek_skip_nl(self, i: int) -> Token:
        j = i
        while j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value == "\n":
            j += 1
        return self.toks[j] if j < len(self.toks) else self.toks[-1]

    def _cur(self) -> Token:
        if self.pos >= len(self.toks):
            return self.toks[-1]
        return self.toks[self.pos]

    def _eat(self, kind: Optional[str] = None, val: object = None) -> Token:
        t = self._cur()
        if kind and t.kind != kind:
            raise BF2SyntaxError(f"expected {kind}, got {t.kind}", _loc(t))
        if val is not None and t.value != val:
            raise BF2SyntaxError(f"expected {val!r}, got {t.value!r}", _loc(t))
        self.pos += 1
        return t

    def _match(self, kind: str, val: object = None) -> bool:
        t = self._cur()
        if t.kind != kind:
            return False
        if val is not None and t.value != val:
            return False
        self.pos += 1
        return True

    def _expect_sym(self, s: str) -> Token:
        return self._eat("SYMBOL", s)

    def parse_module(self) -> A.Module:
        items: list = []
        while self._cur().kind != "EOF":
            while self._cur().kind == "SYMBOL" and self._cur().value == "\n":
                self.pos += 1
            if self._cur().kind == "EOF":
                break
            items.append(self._parse_top())
        return A.Module(items, _loc(self.toks[0]) if self.toks else SourceLoc(1, 1))

    def _parse_top(self):
        while self._cur().kind == "SYMBOL" and self._cur().value == "\n":
            self.pos += 1
        t = self._cur()
        if t.kind == "KEYWORD" and t.value == "seg":
            return self._parse_seg_decl()
        if t.kind == "KEYWORD" and t.value == "struct":
            return self._parse_struct_decl()
        if t.kind == "KEYWORD" and t.value == "fn":
            return self._parse_fn()
        if t.kind == "KEYWORD" and t.value == "watch":
            return self._parse_watch()
        raise BF2SyntaxError("expected seg, struct, fn, or watch", _loc(t))

    def _parse_seg_decl(self) -> A.SegmentDecl:
        loc = _loc(self._eat("KEYWORD", "seg"))
        name = self._eat("IDENT", None).value
        self._expect_sym("{")
        et = self._parse_type_ref()
        n = 1
        if self._match("SYMBOL", "["):
            n = int(self._eat("INT", None).value)
            self._expect_sym("]")
        self._expect_sym("}")
        return A.SegmentDecl(name, et, int(n), loc)

    def _parse_struct_decl(self) -> A.StructDecl:
        loc = _loc(self._eat("KEYWORD", "struct"))
        nm = self._eat("IDENT", None).value
        self._expect_sym("{")
        fields: list[tuple[str, A.TypeRef]] = []
        while self._cur().kind != "EOF" and not (self._cur().kind == "SYMBOL" and self._cur().value == "}"):
            fn = self._eat("IDENT", None).value
            self._expect_sym(":")
            fields.append((fn, self._parse_type_ref()))
            if self._cur().kind == "SYMBOL" and self._cur().value == ",":
                self._eat("SYMBOL", ",")
        self._expect_sym("}")
        return A.StructDecl(nm, fields, loc)

    def _parse_fn(self) -> A.FunctionDef:
        loc = _loc(self._eat("KEYWORD", "fn"))
        nm = self._eat("IDENT", None).value
        self._expect_sym("(")
        params: list[tuple[str, A.TypeRef]] = []
        if not self._match("SYMBOL", ")"):
            while True:
                pn = self._eat("IDENT", None).value
                self._expect_sym(":")
                pt = self._parse_type_ref()
                params.append((pn, pt))
                if self._match("SYMBOL", ")"):
                    break
                self._expect_sym(",")
        self._eat("SYMBOL", "-")
        self._expect_sym(">")
        rt = self._parse_type_ref()
        body = self._parse_block()
        return A.FunctionDef(nm, params, rt, body, loc)

    def _parse_watch(self) -> A.ReactorDef:
        loc = _loc(self._eat("KEYWORD", "watch"))
        tgt = self._parse_ref_expr()
        body = self._parse_block()
        return A.ReactorDef(tgt, body, loc)

    def _parse_block(self) -> A.Block:
        loc = _loc(self._expect_sym("{"))
        stmts: list = []
        while True:
            self._skip_nl()
            if self._match("SYMBOL", "}"):
                break
            stmts.append(self._parse_stmt())
        return A.Block(stmts, loc)

    def _parse_stmt(self):
        self._skip_nl()
        t = self._cur()
        loc = _loc(t)
        if t.kind == "KEYWORD" and t.value == "seg":
            d = self._parse_seg_decl()
            return A.SegmentStmt(d, d.loc)
        if t.kind == "KEYWORD" and t.value == "struct":
            d = self._parse_struct_decl()
            return A.StructStmt(d, d.loc)
        if t.kind == "KEYWORD" and t.value == "if":
            return self._parse_if()
        if t.kind == "KEYWORD" and t.value == "load":
            self._eat("KEYWORD", "load")
            r = self._parse_ref_expr()
            return A.LoadOp(r, loc)
        if t.kind == "KEYWORD" and t.value == "store":
            self._eat("KEYWORD", "store")
            r = self._parse_ref_expr()
            return A.StoreOp(r, loc)
        if t.kind == "KEYWORD" and t.value == "swap":
            self._eat("KEYWORD", "swap")
            r = self._parse_ref_expr()
            return A.SwapOp(r, loc)
        if t.kind == "KEYWORD" and t.value == "label":
            self._eat("KEYWORD", "label")
            nm = self._eat("IDENT", None).value
            return A.LabelStmt(nm, loc)
        if t.kind == "KEYWORD" and t.value == "jump":
            self._eat("KEYWORD", "jump")
            nm = self._eat("IDENT", None).value
            return A.JumpStmt(nm, loc)
        if t.kind == "KEYWORD" and t.value == "ret":
            self._eat("KEYWORD", "ret")
            self._skip_nl()
            if self._cur().kind == "SYMBOL" and self._cur().value == "}":
                return A.RetStmt(None, loc)
            e = self._parse_expr()
            return A.RetStmt(e, loc)
        if t.kind == "KEYWORD" and t.value == "do":
            loc = _loc(self._eat("KEYWORD", "do"))
            ex = self._parse_expr()
            return A.ExprStmt(ex, loc)
        if t.kind == "KEYWORD" and t.value == "ptrread":
            loc = _loc(self._eat("KEYWORD", "ptrread"))
            nm = self._eat("IDENT", None).value
            return A.PtrRead(nm, loc)
        if t.kind == "KEYWORD" and t.value == "alloc":
            return self._parse_alloc()
        if t.kind == "KEYWORD" and t.value == "free":
            self._eat("KEYWORD", "free")
            p = self._eat("IDENT", None).value
            return A.FreeStmt(p, loc)
        if t.kind == "KEYWORD" and t.value == "call":
            c = self._parse_call_expr()
            return A.CallStmt(c, loc)
        if t.kind == "TYPE" or (t.kind == "KEYWORD" and t.value == "ptr"):
            ty = self._parse_type_ref()
            nm = self._eat("IDENT", None).value
            if self._match("SYMBOL", "="):
                init = self._parse_expr()
                if isinstance(ty, A.TypeRef) and ty.name == "ptr":
                    return A.PtrDecl(ty.inner or A.TypeRef("i8"), nm, init, loc)
                return A.VarDecl(ty, nm, init, loc)
            raise BF2SyntaxError("expected = in declaration", loc)
        if (
            t.kind == "IDENT"
            and self.pos + 2 < len(self.toks)
            and self.toks[self.pos + 1].kind == "IDENT"
            and self.toks[self.pos + 2].kind == "SYMBOL"
            and self.toks[self.pos + 2].value == "="
        ):
            ty = A.TypeRef(str(t.value))
            self._eat("IDENT", None)
            nm = self._eat("IDENT", None).value
            self._expect_sym("=")
            init = self._parse_expr()
            return A.VarDecl(ty, nm, init, loc)
        if t.kind == "SYMBOL" and t.value == "[":
            self._eat("SYMBOL", "[")
            return A.LoopBF(self._parse_block(), loc)
        if t.kind == "SYMBOL" and t.value == "{":
            self._eat("SYMBOL", "{")
            cnt = self._eat("INT", None).value
            self._expect_sym("}")
            return A.LoopCounted(int(cnt), self._parse_block(), loc)
        if t.kind == "SYMBOL" and t.value == ".":
            self._eat("SYMBOL", ".")
            if self._cur().kind == "IDENT" and self._cur().value in ("i", "f", "s"):
                return self._parse_io_dot_rest(loc)
            return A.IOStmt(".", None, loc)
        if t.kind == "SYMBOL" and t.value == ",":
            self._eat("SYMBOL", ",")
            if self._match("KEYWORD", "s"):
                return A.IOStmt(",s", None, loc)
            return A.IOStmt(",", None, loc)
        if t.kind == "SYMBOL" and t.value == "@":
            m = self._parse_move_or_assign()
            return m
        if t.kind == "SYMBOL" and t.value == "*":
            loc = _loc(self._eat("SYMBOL", "*"))
            nm = self._eat("IDENT", None).value
            self._skip_nl()
            if self._match("SYMBOL", "="):
                rhs = self._parse_expr()
                return A.PtrWrite(nm, rhs, loc)
            return A.PtrRead(nm, loc)
        if t.kind == "IDENT":
            nm = str(t.value)
            j = self.pos + 1
            while j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value == "\n":
                j += 1
            if (
                j < len(self.toks)
                and self.toks[j].kind == "KEYWORD"
                and self.toks[j].value == "ptr"
            ):
                k = j + 1
                while k < len(self.toks) and self.toks[k].kind == "SYMBOL" and self.toks[k].value == "\n":
                    k += 1
                if k < len(self.toks) and self.toks[k].kind == "SYMBOL" and self.toks[k].value in "+-":
                    op = self.toks[k].value
                    m = k + 1
                    while m < len(self.toks) and self.toks[m].kind == "SYMBOL" and self.toks[m].value == "\n":
                        m += 1
                    if m < len(self.toks) and self.toks[m].kind == "INT":
                        self._eat("IDENT", None)
                        self._eat("KEYWORD", "ptr")
                        self._eat("SYMBOL", op)
                        dv = int(self._eat("INT", None).value)
                        if op == "-":
                            dv = -dv
                        return A.PtrArith(nm, dv, loc)
            nxt = self._peek_skip_nl(self.pos + 1)
            if nxt.kind == "SYMBOL" and nxt.value == "(":
                self._eat("IDENT", None)
                return self._parse_call_after_name(nm, loc)
            j = self.pos + 1
            while j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value == "\n":
                j += 1
            if j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value == ".":
                k = j + 1
                while k < len(self.toks) and self.toks[k].kind == "SYMBOL" and self.toks[k].value == "\n":
                    k += 1
                if (
                    k < len(self.toks)
                    and self.toks[k].kind == "IDENT"
                    and self.toks[k].value in ("i", "f", "s")
                ):
                    self._eat("IDENT", None)
                    self._skip_nl()
                    self._expect_sym(".")
                    suf = self._eat("IDENT", None).value
                    return A.IOStmt("." + suf, A.Ident(nm, loc), loc)
            if nxt.kind == "SYMBOL" and nxt.value in ("[", ".", "="):
                r = self._parse_ref_expr()
                self._skip_nl()
                if self._cur().kind == "SYMBOL" and self._cur().value in "+-*/":
                    op = self._eat("SYMBOL", None).value
                    if self._cur().kind == "INT":
                        amt: Union[int, float] = int(self._eat("INT", None).value)
                    elif self._cur().kind == "FLOAT":
                        amt = float(self._eat("FLOAT", None).value)
                    elif op in "+-":
                        amt = 1
                    else:
                        raise BF2SyntaxError("expected number", loc)
                    return A.CellArithRef(r, op, amt, loc)
                if self._match("SYMBOL", "="):
                    rhs = self._parse_expr()
                    return A.AssignStmt(r, rhs, loc)
                raise BF2SyntaxError("expected = or arith", loc)
            raise BF2SyntaxError("invalid statement", loc)
        if t.kind == "SYMBOL" and t.value in "><":
            d = 1 if t.value == ">" else -1
            self._eat("SYMBOL", None)
            if self._cur().kind == "INT":
                n = int(self._eat("INT", None).value)
                d *= max(1, n)
            return A.MoveRel(d, loc)
        if t.kind == "SYMBOL" and t.value in "+-*/":
            op = t.value
            self._eat("SYMBOL", None)
            if self._cur().kind == "INT":
                n = self._eat("INT", None).value
                return A.CellArith(op, int(n), loc)
            if self._cur().kind == "FLOAT":
                n = self._eat("FLOAT", None).value
                return A.CellArith(op, float(n), loc)
            if op in "+-":
                return A.CellArith(op, 1, loc)
            raise BF2SyntaxError("expected number after operator", loc)
        if t.kind == "SYMBOL" and t.value == "=":
            self._eat("SYMBOL", "=")
            v = self._parse_literal_for_cell()
            return A.CellAssignLit(v, loc)
        if t.kind == "KEYWORD" and t.value == "ptr":
            raise BF2SyntaxError("ptr must be in type position", loc)
        raise BF2SyntaxError(f"unexpected token {t.kind} {t.value!r}", loc)

    def _parse_literal_for_cell(self) -> Union[int, float, bool]:
        t = self._cur()
        if t.kind == "INT":
            self._eat("INT", None)
            return int(t.value)
        if t.kind == "FLOAT":
            self._eat("FLOAT", None)
            return float(t.value)
        if t.kind == "KEYWORD" and t.value in ("true", "false"):
            self._eat("KEYWORD", None)
            return t.value == "true"
        raise BF2SyntaxError("literal expected", _loc(t))

    def _parse_move_or_assign(self):
        loc = _loc(self._eat("SYMBOL", "@"))
        r = self._parse_ref_expr()
        return A.MoveOp(r, loc)

    def _parse_if(self) -> A.IfStmt:
        loc = _loc(self._eat("KEYWORD", "if"))
        self._expect_sym("(")
        cond = self._parse_cond()
        self._expect_sym(")")
        th = self._parse_block()
        el = self._parse_block() if self._match("KEYWORD", "else") else None
        return A.IfStmt(cond, th, el, loc)

    def _parse_cond(self) -> A.Cond:
        t = self._cur()
        if self._match("SYMBOL", ">"):
            n = int(self._eat("INT", None).value)
            if n == 0:
                return A.Cond(">0")
            return A.Cond(">N", n)
        if self._match("SYMBOL", "<"):
            n = int(self._eat("INT", None).value)
            if n == 0:
                return A.Cond("<0")
            return A.Cond("<N", n)
        if self._match("SYMBOL", "=") and self._match("SYMBOL", "="):
            n = int(self._eat("INT", None).value)
            if n == 0:
                return A.Cond("==0")
            return A.Cond("==N", n)
        if self._match("SYMBOL", "!"):
            self._expect_sym("=")
            n = int(self._eat("INT", None).value)
            if n == 0:
                return A.Cond("!=0")
            raise BF2SyntaxError("bad condition", _loc(t))
        raise BF2SyntaxError("bad condition", _loc(t))

    def _parse_alloc(self) -> A.AllocStmt:
        loc = _loc(self._eat("KEYWORD", "alloc"))
        ty = self._parse_type_ref()
        n = int(self._eat("INT", None).value)
        nm = ""
        if self._match("KEYWORD", "as"):
            nm = self._eat("IDENT", None).value
        return A.AllocStmt(ty, int(n), nm, loc)

    def _parse_io_dot_rest(self, loc: SourceLoc) -> A.IOStmt:
        suf = self._eat("IDENT", None).value
        if suf not in ("i", "f", "s"):
            raise BF2SyntaxError("expected i, f, or s after .", loc)
        return A.IOStmt("." + suf, None, loc)

    def _parse_type_ref(self) -> A.TypeRef:
        t = self._cur()
        if t.kind == "KEYWORD" and t.value == "ptr":
            self._eat("KEYWORD", "ptr")
            self._expect_sym("<")
            inner = self._parse_type_ref()
            self._expect_sym(">")
            return A.TypeRef("ptr", inner)
        if t.kind == "TYPE":
            self._eat("TYPE", None)
            return A.TypeRef(str(t.value))
        if t.kind == "IDENT":
            self._eat("IDENT", None)
            return A.TypeRef(str(t.value))
        raise BF2SyntaxError("type expected", _loc(t))

    def _parse_call_after_name(self, nm: str, loc: SourceLoc) -> A.CallStmt:
        self._expect_sym("(")
        args: list[A.Expr] = []
        if not self._match("SYMBOL", ")"):
            while True:
                args.append(self._parse_expr())
                if self._match("SYMBOL", ")"):
                    break
                self._expect_sym(",")
        return A.CallStmt(A.Call(nm, args, loc), loc)

    def _parse_ref_expr(self) -> A.RefExpr:
        loc = _loc(self._cur())
        parts: list = []
        if self._match("SYMBOL", "@"):
            parts.append("@")
            parts.append(self._eat("IDENT", None).value)
            return A.RefExpr(parts, loc)
        head = self._eat("IDENT", None).value
        parts.append(head)
        while True:
            if self._match("SYMBOL", "["):
                e = self._parse_expr()
                self._expect_sym("]")
                parts.append(e)
            elif self._match("SYMBOL", "."):
                parts.append(self._eat("IDENT", None).value)
            else:
                break
        return A.RefExpr(parts, loc)

    def _parse_expr(self) -> A.Expr:
        return self._parse_add()

    def _parse_add(self) -> A.Expr:
        e = self._parse_mul()
        while self._cur().kind == "SYMBOL" and self._cur().value in "+-":
            op = self._eat("SYMBOL", None).value
            r = self._parse_mul()
            e = A.BinOp(op, e, r, _loc(self.toks[max(0, self.pos - 1)]))
        return e

    def _parse_mul(self) -> A.Expr:
        e = self._parse_unary()
        while self._cur().kind == "SYMBOL" and self._cur().value in "*/":
            op = self._eat("SYMBOL", None).value
            r = self._parse_unary()
            e = A.BinOp(op, e, r, _loc(self.toks[max(0, self.pos - 1)]))
        return e

    def _parse_unary(self) -> A.Expr:
        t = self._cur()
        loc = _loc(t)
        if t.kind == "SYMBOL" and t.value == "-":
            self._eat("SYMBOL", "-")
            return A.Unary("-", self._parse_unary(), loc)
        if t.kind == "SYMBOL" and t.value == "*":
            self._eat("SYMBOL", "*")
            nm = self._eat("IDENT", None).value
            return A.Unary("*", A.Ident(nm, loc), loc)
        if t.kind == "SYMBOL" and t.value == "&":
            self._eat("SYMBOL", "&")
            r = self._parse_ref_expr()
            return A.Unary("&", r, loc)
        return self._parse_primary()

    def _parse_primary(self) -> A.Expr:
        t = self._cur()
        loc = _loc(t)
        if t.kind == "INT":
            self._eat("INT", None)
            return A.IntLit(int(t.value), loc)
        if t.kind == "FLOAT":
            self._eat("FLOAT", None)
            return A.FloatLit(float(t.value), loc)
        if t.kind == "KEYWORD" and t.value in ("true", "false"):
            self._eat("KEYWORD", None)
            return A.BoolLit(t.value == "true", loc)
        if t.kind == "KEYWORD" and t.value == "call":
            return self._parse_call_expr()
        if t.kind == "IDENT":
            if self._peek_call_args():
                return self._parse_call_expr()
            if self._peek_ref():
                return self._parse_ref_expr()
            self._eat("IDENT", None)
            return A.Ident(str(t.value), loc)
        if t.kind == "SYMBOL" and t.value == "(":
            self._eat("SYMBOL", "(")
            e = self._parse_expr()
            self._expect_sym(")")
            return e
        raise BF2SyntaxError("expression expected", loc)

    def _peek_call_args(self) -> bool:
        if self._cur().kind != "IDENT":
            return False
        j = self.pos + 1
        while j < len(self.toks) and self.toks[j].kind in ("SYMBOL",) and self.toks[j].value == "\n":
            j += 1
        return j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value == "("

    def _peek_ref(self) -> bool:
        if self._cur().kind != "IDENT":
            return False
        j = self.pos + 1
        while j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value == "\n":
            j += 1
        return j < len(self.toks) and self.toks[j].kind == "SYMBOL" and self.toks[j].value in ("[", ".")

    def _parse_call_expr(self) -> A.Call:
        loc = _loc(self._cur())
        if self._match("KEYWORD", "call"):
            nm = self._eat("IDENT", None).value
        else:
            nm = self._eat("IDENT", None).value
        self._expect_sym("(")
        args: list[A.Expr] = []
        if not self._match("SYMBOL", ")"):
            while True:
                args.append(self._parse_expr())
                if self._match("SYMBOL", ")"):
                    break
                self._expect_sym(",")
        return A.Call(nm, args, loc)


def parse_tokens(tokens: List[Token], text: str) -> A.Module:
    p = Parser(tokens, text)
    return p.parse_module()


def tokenize_all(src: str) -> List[Token]:
    return list(Lexer(src).tokenize())


def parse_source(text: str, *, use_linux_stdlib: bool = False) -> A.Module:
    from bf2.errors import BF2SyntaxError

    try:
        mod = parse_tokens(tokenize_all(text), text)
        mod.use_linux_stdlib = use_linux_stdlib
        return mod
    except BF2SyntaxError:
        raise
    except Exception as e:
        raise BF2SyntaxError(str(e)) from e
