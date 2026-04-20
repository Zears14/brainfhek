# BF2 — Brainfhek (aka Brainfuck 2.0)

Python interpreter for **BF2**: reactive `watch` cells and typed `seg`/`struct` memory, plus **`python -m bf2 compile`** to emit standalone `.ll` (libc only; see [LLVM_NOTES.md](LLVM_NOTES.md)).

## Install

```bash
cd /path/to/bfck-llvm
pip install -e .
```

Or without install:

```bash
PYTHONPATH=. python -m bf2 run examples/fibonacci.bf2
```

## Run

```bash
python -m bf2 run path/to/file.bf2
python -m bf2 compile path/to/file.bf2 -o out.ll   # then: clang out.ll -o a.out
```

Most programs under `examples/` compile with `bf2 compile` and link with `clang` (libc only). Classic-only `.bf` sources (e.g. `hello.bf` containing only `><+-.,[]`) are meant for `bf2 run`, not the BF2 compiler.

- Programs that contain **only** classic Brainfuck (`><+-.,[]`) plus whitespace are executed on a 30 000-cell `i8` tape (no `seg` needed).
- BF2 programs are parsed, type-checked, and interpreted.

## Layout

| Path | Purpose |
|------|---------|
| `bf2/lexer.py` | `Token` stream |
| `bf2/parser.py` | Recursive descent → AST |
| `bf2/ast_nodes.py` | AST + `to_llvm_ir_hint()` |
| `bf2/typechecker.py` | Basic type / symbol checks |
| `bf2/memory.py` | Segments, struct layout, `Pointer` |
| `bf2/reactor.py` | `ReactorEngine` (no recursive fire) |
| `bf2/interpreter.py` | Evaluation |
| `bf2/stdlib/stdlib.bf2` | Linux libc prelude (`#include "stdlib"`) |
| `examples/` | Sample programs |
| `SPEC.md` | Language + LLVM table |

## Cheat sheet

| BF2 | Meaning |
|-----|---------|
| `seg s { i32[16] }` | Typed array segment |
| `struct T { x: i32, y: i32 }` | Struct type |
| `watch s[0] { ... }` | Reactor after writes to that cell |
| `fn f(a: i32) -> i32 { ret a }` | Function |
| `if(== 0) { ... }` | Branch on **current cell** vs literal |
| `load s[0]` / `store s[0]` | Cell ↔ address |
| `.` / `,` / `.i` / `.f` | I/O |

## Tests

```bash
pip install pytest
pytest tests/
```
