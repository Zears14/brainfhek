from __future__ import annotations

from typing import Any, List, Optional, Union, Tuple

from bf2.core import ast as A
from bf2.core.errors import BF2SyntaxError, SourceLoc
from bf2.compiler.lexer import Token, Lexer

def parse_source(source: str, use_linux_stdlib: bool = False) -> A.Module:
    tokens = list(Lexer(source).tokenize())
    mod = Parser(tokens).parse()
    mod.use_linux_stdlib = use_linux_stdlib
    return mod

def parse_tokens(tokens: List[Token], use_linux_stdlib: bool = False) -> A.Module:
    mod = Parser(tokens).parse()
    mod.use_linux_stdlib = use_linux_stdlib
    return mod

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> A.Module:
        items = []
        while not self._is_at_end():
            while self._match("SYMBOL", "\n"):
                pass
            if self._is_at_end():
                break
            node = self._parse_top_level()
            if node:
                items.append(node)
        return A.Module(items, SourceLoc(1, 1))

    def _parse_top_level(self) -> Any:
        if self._match("KEYWORD", "fn"):
            return self._parse_function()
        if self._match("KEYWORD", "seg"):
            return self._parse_segment(reactive=False)
        if self._match("KEYWORD", "seglink"):
            return self._parse_segment(reactive=True)
        if self._match("KEYWORD", "struct"):
            return self._parse_struct()
        if self._match("KEYWORD", "watch"):
            return self._parse_reactor()
        if self._match("KEYWORD", "#include"):
            self._consume("STRING", "include path")
            return None
        raise BF2SyntaxError(f"Unexpected top-level token: {self._peek()}", self._loc())

    def _parse_function(self) -> A.FunctionDef:
        loc = self._loc()
        name = self._consume("IDENT", "fn name").value
        self._consume("SYMBOL", "(")
        params: List[Tuple[str, A.TypeRef]] = []
        if not self._check("SYMBOL", ")"):
            while True:
                pname = self._consume("IDENT", "param name").value
                self._consume("SYMBOL", ":")
                ty = self._parse_type()
                params.append((pname, ty))
                if not self._match("SYMBOL", ","):
                    break
        self._consume("SYMBOL", ")")
        self._consume("SYMBOL", "->")
        ret = self._parse_type()
        return A.FunctionDef(name, params, ret, self._parse_block(), loc)

    def _parse_segment(self, reactive: bool = False) -> A.SegmentDecl:
        loc = self._loc()
        name = self._consume("IDENT", "seg name").value
        self._consume("SYMBOL", "{")
        ty = self._parse_type()
        n = 1
        if self._match("SYMBOL", "["):
            n = int(self._consume("INT", "seg length").value)
            self._consume("SYMBOL", "]")
        self._consume("SYMBOL", "}")
        init = None
        if self._match("SYMBOL", "="):
            init = self._parse_expression()
        elif reactive:
            raise BF2SyntaxError(f"seglink '{name}' must have an initial expression", loc)
        return A.SegmentDecl(name, ty, n, init, reactive, loc)

    def _parse_struct(self) -> A.StructDecl:
        loc = self._loc()
        name = self._consume("IDENT", "struct name").value
        self._consume("SYMBOL", "{")
        fields: List[Tuple[str, A.TypeRef]] = []
        while not self._check("SYMBOL", "}"):
            pname = self._consume("IDENT", "field name").value
            self._consume("SYMBOL", ":")
            fields.append((pname, self._parse_type()))
            self._match("SYMBOL", ",")
        self._consume("SYMBOL", "}")
        return A.StructDecl(name, fields, loc)

    def _parse_reactor(self) -> A.ReactorDef:
        loc = self._loc()
        tgt = self._parse_ref_expr()
        return A.ReactorDef(tgt, self._parse_block(), loc)

    def _parse_block(self) -> A.Block:
        loc = self._loc()
        self._consume("SYMBOL", "{")
        stmts = []
        while not self._check("SYMBOL", "}") and not self._is_at_end():
            while self._match("SYMBOL", "\n"):
                pass
            if self._check("SYMBOL", "}"):
                break
            stmts.append(self._parse_statement())
        self._consume("SYMBOL", "}")
        return A.Block(stmts, loc)

    def _parse_statement(self) -> A.ASTNode:
        while self._match("SYMBOL", "\n"):
            pass
        loc = self._loc()
        t = self._peek()
        if t.kind == "KEYWORD":
            if t.value == "if":
                return self._parse_if()
            if t.value == "do":
                self._adv()
                return A.ExprStmt(self._parse_expression(), loc)
            if t.value == "ret":
                self._adv()
                return A.RetStmt(self._parse_expression() if not self._check("SYMBOL", "}") else None, loc)
            if t.value == "call":
                self._adv()
                return A.CallStmt(self._parse_call_expr(), loc)
            if t.value == "ptr":
                return self._parse_maybe_ptr_stmt()
            if t.value == "seg":
                self._adv()
                d = self._parse_segment()
                return A.SegmentStmt(d, d.loc)
            if t.value == "struct":
                self._adv()
                d = self._parse_struct()
                return A.StructStmt(d, d.loc)
            if t.value == "load":
                self._adv()
                return A.LoadOp(self._parse_ref_expr(), loc)
            if t.value == "store":
                self._adv()
                return A.StoreOp(self._parse_ref_expr(), loc)
            if t.value == "swap":
                self._adv()
                return A.SwapOp(self._parse_ref_expr(), loc)
            if t.value == "label":
                self._adv()
                return A.LabelStmt(self._consume("IDENT", "label").value, loc)
            if t.value == "jump":
                self._adv()
                return A.JumpStmt(self._consume("IDENT", "target").value, loc)
            if t.value == "alloc":
                return self._parse_alloc()
            if t.value == "free":
                self._adv()
                return A.FreeStmt(self._consume("IDENT", "ptr").value, loc)
            if t.value == "unlink":
                self._adv()
                return A.UnlinkStmt(self._consume("IDENT", "target").value, loc)
            if t.value == "ptrread":
                self._adv()
                return A.PtrRead(self._consume("IDENT", "ptr").value, loc)
        if t.kind == "TYPE":
            return self._parse_decl_stmt()
        if t.kind == "SYMBOL":
            if t.value == "@":
                self._adv()
                return A.MoveOp(self._parse_ref_expr(), loc)
            if t.value == "[":
                self._adv()
                body = self._parse_block()
                self._consume("SYMBOL", "]")
                return A.LoopBF(body, loc)
            if t.value == "{":
                return self._parse_loop_counted()
            if t.value == ".":
                return self._parse_io_dot()
            if t.value == ",":
                return self._parse_io_comma()
            if t.value == "*":
                return self._parse_ptr_write_read()
            if t.value == "=":
                self._adv()
                return A.CellAssignLit(self._parse_literal(), loc)
            if t.value in "+-*/":
                self._adv()
                n = self._parse_arith_amount()
                return A.CellArith(t.value, n if n is not None else 1, loc)
            if t.value in "><":
                self._adv()
                n = 1
                if self._check("INT"):
                    n = int(self._consume("INT", "move amount").value)
                return A.MoveRel(n if t.value == ">" else -n, loc)
        if t.kind in ("IDENT", "STRING", "INT", "FLOAT"):
            if t.kind == "IDENT":
                if self._peek_decl():
                    return self._parse_decl_stmt()
                if self._peek_ptr_arith():
                    return self._parse_ptr_arith()
            
            # 1. Try parsing a RefExpr (for Assign/Arith/IO)
            # or a literal (for IO/ExprStmt)
            expr = None
            if t.kind == "IDENT" or (t.kind == "SYMBOL" and t.value == "@"):
                expr = self._parse_ref_expr()
            else:
                expr = self._parse_primary()
            
            # Now we have the 'head' of the statement. Check for operators.
            t_next = self._peek()
            
            # IO Suffix: expr .ident OR expr .i64
            if t_next.value == "." and self._peek(1).value in ("i", "f", "s", "ir", "fr", "i64", "i64r", "idx"):
                suf = self._peek(1).value
                if suf in ("i", "f", "s", "ir", "fr", "i64", "i64r", "idx"):
                    self._consume("SYMBOL", ".")
                    self._adv() # consume suffix
                    return A.IOStmt("." + str(suf), expr, loc)

            # Side-effects for RefExpr: re = expr, re + amount
            if isinstance(expr, A.RefExpr):
                if t_next.value == "=":
                    self._adv()
                    return A.AssignStmt(expr, self._parse_expression(), loc)
                if t_next.value in "+-*/":
                    op = self._consume("SYMBOL", "op").value
                    return A.CellArithRef(expr, op, self._parse_arith_amount(), loc)
            
            # If nothing matched, it might be part of a larger expression?
            # E.g. (1) + 2
            # But wait, we already consumed the primary. 
            # If we want to support ExprStmt fully, we should continue parsing.
            # But in BF2, standalone expressions are rare except for side-effects.
            
            # Let's check if we can continue parsing this as an expression
            if t_next.value in "+-*/==!=><":
                # It's a complex expression, parse the rest
                # (This is a bit hacky because we don't have a partial expr parser easily)
                # But we can try to 'undo' the parse or just handle it.
                # Actually, the most common case for ExprStmt is a Call.
                pass
            
            return A.ExprStmt(expr, loc)
        raise BF2SyntaxError(f"Unexpected statement: {t}", loc)

    def _parse_if(self) -> A.IfStmt:
        loc = self._loc()
        self._adv()
        self._consume("SYMBOL", "(")
        cond = self._parse_cond()
        self._consume("SYMBOL", ")")
        th = self._parse_block()
        el = self._parse_block() if self._match("KEYWORD", "else") else None
        return A.IfStmt(cond, th, el, loc)

    def _parse_maybe_ptr_stmt(self) -> A.ASTNode:
        # Could be ptr<T> x = ...
        if self._peek(1).value == "<":
            return self._parse_decl_stmt()
        raise BF2SyntaxError("unexpected ptr keyword in statement", self._loc())

    def _parse_decl_stmt(self) -> A.VarDecl | A.PtrDecl:
        loc = self._loc()
        ty = self._parse_type()
        name = self._consume("IDENT", "var name").value
        self._consume("SYMBOL", "=")
        init = self._parse_expression()
        if ty.name == "ptr":
            return A.PtrDecl(ty.inner or A.TypeRef("i8"), name, init, loc)
        return A.VarDecl(ty, name, init, loc)

    def _parse_loop_counted(self) -> A.LoopCounted:
        loc = self._loc()
        self._adv()
        n = int(self._consume("INT", "count").value)
        self._consume("SYMBOL", "}")
        return A.LoopCounted(n, self._parse_block(), loc)

    def _parse_io_dot(self) -> A.IOStmt | A.Expr:
        loc = self._loc()
        self._adv()
        t = self._peek()
        if t.kind == "IDENT":
            self._adv()
            return A.IOStmt("." + str(t.value), None, loc)
        return A.IOStmt(".", None, loc)

    def _parse_io_comma(self) -> A.IOStmt:
        loc = self._loc()
        self._adv()
        return A.IOStmt(",s" if self._match("KEYWORD", "s") else ",", None, loc)

    def _parse_ptr_write_read(self) -> A.PtrWrite | A.PtrRead:
        loc = self._loc()
        self._adv()
        name = self._consume("IDENT", "ptr name").value
        if self._match("SYMBOL", "="):
            return A.PtrWrite(name, self._parse_expression(), loc)
        return A.PtrRead(name, loc)

    def _parse_ptr_arith(self) -> A.PtrArith:
        loc = self._loc()
        name = self._consume("IDENT", "ptr name").value
        self._consume("KEYWORD", "ptr")
        op = self._consume("SYMBOL", "+ or -").value
        delta = int(self._consume("INT", "delta").value)
        return A.PtrArith(name, delta if op == "+" else -delta, loc)

    def _parse_alloc(self) -> A.AllocStmt:
        loc = self._loc()
        self._adv()
        ty = self._parse_type()
        n = int(self._consume("INT", "size").value)
        nm = self._consume("IDENT", "as name").value if self._match("KEYWORD", "as") else ""
        return A.AllocStmt(ty, n, nm, loc)

    def _parse_expression(self) -> A.Expr:
        return self._parse_or()

    def _parse_or(self) -> A.Expr:
        e = self._parse_xor()
        while self._peek().value == "|":
            op = self._consume("SYMBOL", "op").value
            e = A.BinOp(str(op), e, self._parse_xor(), e.loc)
        return e

    def _parse_xor(self) -> A.Expr:
        e = self._parse_and()
        while self._peek().value == "^":
            op = self._consume("SYMBOL", "op").value
            e = A.BinOp(str(op), e, self._parse_and(), e.loc)
        return e

    def _parse_and(self) -> A.Expr:
        e = self._parse_shift()
        while self._peek().value == "&":
            op = self._consume("SYMBOL", "op").value
            e = A.BinOp(str(op), e, self._parse_shift(), e.loc)
        return e

    def _parse_shift(self) -> A.Expr:
        e = self._parse_add()
        while True:
            t1 = self._peek()
            t2 = self._peek(1)
            if t1.value == "<" and t2.value == "<":
                self._adv(2)
                e = A.BinOp("<<", e, self._parse_add(), e.loc)
            elif t1.value == ">" and t2.value == ">":
                self._adv(2)
                e = A.BinOp(">>", e, self._parse_add(), e.loc)
            else:
                break
        return e

    def _parse_add(self) -> A.Expr:
        e = self._parse_mul()
        while self._peek().value in "+-":
            op = self._consume("SYMBOL", "op").value
            e = A.BinOp(op, e, self._parse_mul(), e.loc)
        return e

    def _parse_mul(self) -> A.Expr:
        e = self._parse_unary()
        while self._peek().value in "*/":
            op = self._consume("SYMBOL", "op").value
            e = A.BinOp(op, e, self._parse_unary(), e.loc)
        return e

    def _parse_unary(self) -> A.Expr:
        loc = self._loc()
        t = self._peek()
        if t.kind == "SYMBOL":
            if t.value == "-":
                self._adv()
                return A.Unary("-", self._parse_unary(), loc)
            if t.value == "*":
                self._adv()
                return A.Unary("*", A.Ident(self._consume("IDENT", "deref").value, loc), loc)
            if t.value == "&":
                self._adv()
                return A.Unary("&", self._parse_ref_expr(), loc)
        return self._parse_primary()

    def _parse_primary(self) -> A.Expr:
        loc = self._loc()
        t = self._peek()
        if t.kind == "INT":
            self._adv()
            return A.IntLit(int(t.value), loc)
        if t.kind == "FLOAT":
            self._adv()
            return A.FloatLit(float(t.value), loc)
        if t.kind == "STRING":
            self._adv()
            return A.StringLit(str(t.value), loc)
        if t.kind == "KEYWORD" and t.value in ("true", "false"):
            self._adv()
            return A.BoolLit(t.value == "true", loc)
        if t.kind == "KEYWORD" and t.value == "call":
            return self._parse_call_expr()
        if t.kind == "IDENT":
            if self._peek_call():
                return self._parse_call_expr()
            if self._peek_ref():
                return self._parse_ref_expr()
            self._adv()
            return A.Ident(t.value, loc)
        if t.kind == "SYMBOL" and t.value == "(":
            self._adv()
            e = self._parse_expression()
            self._consume("SYMBOL", ")")
            return e
        if t.kind == "SYMBOL" and t.value == "@":
            return self._parse_ref_expr()
        raise BF2SyntaxError(f"Expression expected, got {t}", loc)

    def _parse_call_expr(self) -> A.Call:
        loc = self._loc()
        nm = self._consume("IDENT", "call name").value if not self._match("KEYWORD", "call") else self._consume("IDENT", "call name").value
        self._consume("SYMBOL", "(")
        args = []
        if not self._check("SYMBOL", ")"):
            while True:
                args.append(self._parse_expression())
                if self._match("SYMBOL", ")"):
                    break
                self._consume("SYMBOL", ",")
        else:
            self._adv()
        return A.Call(nm, args, loc)

    def _parse_ref_expr(self) -> A.RefExpr:
        loc = self._loc()
        parts = []
        if self._match("SYMBOL", "@"):
            parts.append("@")
            parts.append(self._consume("IDENT", "seg").value)
            return A.RefExpr(parts, loc)
        parts.append(self._consume("IDENT", "head").value)
        while True:
            if self._match("SYMBOL", "["):
                parts.append(self._parse_expression())
                self._consume("SYMBOL", "]")
            elif self._match("SYMBOL", "."):
                # Lookahead: if it's .i, .f, or .s, it's an IO statement suffix, not a field.
                nxt = self._peek()
                if nxt.value in ("i", "f", "s", "ir", "fr", "i64", "i64r", "idx"):
                    self._backfill("SYMBOL", ".") # Put back the dot
                    break
                parts.append(self._consume("IDENT", "field").value)
            else:
                break
        return A.RefExpr(parts, loc)

    def _backfill(self, kind: str, value: Any) -> None:
        """Pushes a token back onto the stream."""
        self.pos -= 1

    def _parse_cond(self) -> A.Cond:
        t = self._peek()
        if t.value == ">":
            self._adv()
            n = int(self._consume("INT", "val").value)
            return A.Cond(">0" if n == 0 else ">N", n)
        if t.value == "<":
            self._adv()
            n = int(self._consume("INT", "val").value)
            return A.Cond("<0" if n == 0 else "<N", n)
        if t.value == "==":
            self._adv()
            n = int(self._consume("INT", "val").value)
            return A.Cond("==0" if n == 0 else "==N", n)
        if t.value == "!=":
            self._adv()
            n = int(self._consume("INT", "val").value)
            return A.Cond("!=0" if n == 0 else "!=N", n)
        raise BF2SyntaxError(f"Bad condition: {t}", self._loc())

    def _parse_type(self) -> A.TypeRef:
        t = self._peek()
        if t.kind == "KEYWORD" and t.value == "ptr":
            self._adv()
            self._consume("SYMBOL", "<")
            inner = self._parse_type()
            self._consume("SYMBOL", ">")
            return A.TypeRef("ptr", inner)
        if t.kind in ("TYPE", "IDENT"):
            self._adv()
            return A.TypeRef(str(t.value))
        raise BF2SyntaxError(f"Type expected, got {t}", self._loc())

    def _parse_literal(self) -> Any:
        t = self._peek()
        self._adv()
        if t.kind == "INT":
            return int(t.value)
        if t.kind == "FLOAT":
            return float(t.value)
        if t.kind == "STRING":
            return A.StringLit(str(t.value), t.loc)
        if t.kind == "KEYWORD":
            return t.value == "true"
        raise BF2SyntaxError(f"Literal expected, got {t}", self._loc())

    def _parse_arith_amount(self) -> Optional[Union[int, float]]:
        if self._check("INT"):
            return int(self._consume("INT", "val").value)
        if self._check("FLOAT"):
            return float(self._consume("FLOAT", "val").value)
        return None

    def _peek_decl(self) -> bool:
        if self._peek(1).kind == "IDENT" and self._peek(2).value == "=":
            return True
        return False

    def _peek_ptr_arith(self) -> bool:
        return self._peek(1).value == "ptr"

    def _peek_call(self) -> bool:
        j = self.pos + 1
        while j < len(self.tokens) and self.tokens[j].value == "\n":
            j += 1
        return j < len(self.tokens) and self.tokens[j].value == "("

    def _peek_ref(self) -> bool:
        j = self.pos + 1
        while j < len(self.tokens) and self.tokens[j].value == "\n":
            j += 1
        return j < len(self.tokens) and self.tokens[j].value in ("[", ".")

    def _check(self, kind: str, val: Optional[str] = None) -> bool:
        if self._is_at_end():
            return False
        t = self.tokens[self.pos]
        return t.kind == kind and (val is None or t.value == val)

    def _match(self, kind: str, val: Optional[str] = None) -> bool:
        if self._check(kind, val):
            self.pos += 1
            return True
        return False

    def _consume(self, kind: str, msg: str) -> Token:
        if self._check(kind):
            self.pos += 1
            return self.tokens[self.pos-1]
        raise BF2SyntaxError(f"Expected {kind}, got {self._peek()}: {msg}", self._loc())

    def _adv(self, n: int = 1) -> None:
        self.pos += n

    def _peek(self, n: int = 0) -> Token:
        return self.tokens[self.pos+n] if self.pos+n < len(self.tokens) else self.tokens[-1]

    def _is_at_end(self) -> bool:
        return self.pos >= len(self.tokens) or self.tokens[self.pos].kind == "EOF"

    def _loc(self) -> SourceLoc:
        t = self._peek()
        return SourceLoc(t.line, t.col)
