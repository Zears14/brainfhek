# BF2 Language Extended Backus-Naur Form (EBNF)

This document provides a formal grammar specification for the **BF2** language, based on the recursive descent parser implementation in `bf2/parser.py` and `bf2/lexer.py`.

## Tokens (Lexer)

```ebnf
IDENT     ::= [a-zA-Z_][a-zA-Z0-9_]*
INT       ::= ('0'|'1'|'2'|'3'|'4'|'5'|'6'|'7'|'8'|'9')+
FLOAT     ::= INT '.' INT
STRING    ::= '"' [^"]* '"'
TYPE      ::= 'i8' | 'i16' | 'i32' | 'i64' | 'f32' | 'f64' | 'bool'
KEYWORD   ::= 'fn' | 'ret' | 'call' | 'seg' | 'struct' | 'watch' | 'load' 
            | 'store' | 'swap' | 'label' | 'jump' | 'if' | 'else' | 'alloc' 
            | 'as' | 'free' | 'type' | 'ptr' | 'ptrread' | 'do' | 'true' | 'false'
```

## Top-Level Structure

```ebnf
Module       ::= TopLevel*

TopLevel     ::= SegDecl
               | StructDecl
               | FunctionDef
               | WatchDef
```

## Declarations & Types

```ebnf
TypeRef      ::= 'ptr' '<' TypeRef '>'
               | TYPE
               | IDENT

SegDecl      ::= 'seg' IDENT '{' TypeRef ('[' INT ']')? '}' ('=' Expr)?

StructDecl   ::= 'struct' IDENT '{' StructField* '}'
StructField  ::= IDENT ':' TypeRef ','?

FunctionDef  ::= 'fn' IDENT '(' FuncParams? ')' '->' TypeRef Block
FuncParams   ::= IDENT ':' TypeRef (',' IDENT ':' TypeRef)*

WatchDef     ::= 'watch' RefExpr Block
```

## Blocks & Statements

```ebnf
Block        ::= '{' Stmt* '}'

Stmt         ::= SegDecl
               | StructDecl
               | IfStmt
               | 'load' RefExpr
               | 'store' RefExpr
               | 'swap' RefExpr
               | 'label' IDENT
               | 'jump' IDENT
               | 'ret' Expr?
               | 'do' Expr
               | 'ptrread' IDENT
               | AllocStmt
               | 'free' IDENT
               | CallExpr
               | 'call' CallExpr
               | DeclStmt
               | LoopBFStmt
               | LoopCountedStmt
               | IOStmt
               | MoveOpStmt
               | PtrWriteStmt
               | PtrArithStmt
               | PtrReadStmt
               | AssignStmt
               | MoveRelStmt
               | CellArithStmt
               | CellAssignLitStmt

IfStmt       ::= 'if' '(' Cond ')' Block ('else' Block)?
Cond         ::= '>' INT
               | '<' INT
               | '==' INT
               | '!=' INT

AllocStmt    ::= 'alloc' TypeRef INT ('as' IDENT)?

DeclStmt     ::= (TYPE | 'ptr' '<' TypeRef '>' | IDENT) IDENT '=' Expr

LoopBFStmt   ::= '[' Block ']'
LoopCountedStmt ::= '{' INT '}' Block

IOStmt       ::= '.' ('i' | 'f' | 's')?
               | ',' ('s')?
               | IDENT '.' ('i' | 'f' | 's')

MoveOpStmt   ::= '@' RefExpr
MoveRelStmt  ::= ('>' | '<') INT?

PtrWriteStmt ::= '*' IDENT '=' Expr
PtrReadStmt  ::= '*' IDENT
PtrArithStmt ::= IDENT 'ptr' ('+' | '-') INT

AssignStmt   ::= RefExpr '=' Expr

CellArithStmt ::= RefExpr ('+' | '-' | '*' | '/') (INT | FLOAT | '')
                | ('+' | '-' | '*' | '/') (INT | FLOAT)?

CellAssignLitStmt ::= '=' (INT | FLOAT | 'true' | 'false')
```

## Expressions

```ebnf
RefExpr      ::= '@' IDENT
               | IDENT ('[' Expr ']' | '.' IDENT)*

Expr         ::= AddExpr
AddExpr      ::= MulExpr (('+' | '-') MulExpr)*
MulExpr      ::= UnaryExpr (('*' | '/') UnaryExpr)*

UnaryExpr    ::= '-' UnaryExpr
               | '*' IDENT
               | '&' RefExpr
               | PrimaryExpr

PrimaryExpr  ::= INT
               | FLOAT
               | 'true' | 'false'
               | IDENT
               | '(' Expr ')'
               | CallExpr
               | 'call' CallExpr

CallExpr     ::= IDENT '(' CallArgs? ')'
CallArgs     ::= Expr (',' Expr)*
```

## Parsing Notes

- **Context-Sensitive Operators**: Symbols like `+`, `-`, `*`, `/` operate either universally as AST BinOps or selectively as `CellArithStmt` when parsed structurally as statements following refs or naked operators.
- **Classic Memory Moving**: `<` and `>` parse precisely as relative movements across the continuous array segment.
- **Reference Chaining**: `RefExpr` supports `seg[expr]` bindings and implicit C-like struct dot chaining (e.g., `pt.x`).
- **Pointers**: `*`, `&`, and `ptrread`/`ptr` constructs enable C-level explicit pointer semantics.
