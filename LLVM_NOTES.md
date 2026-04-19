# LLVM lowering

BF2 maps to LLVM IR via `bf2/llvm_emit.py` (`emit_llvm_ir`). The module is **standalone** and links with **`clang file.ll -o a.out`** (libc only — no separate C runtime).

- With **`#include "stdlib"`** (preprocessor): I/O lowers to **`snprintf`**, **`strlen`**, **`write`**, **`read`** plus libc declares from the stdlib prelude; no **`printf`** / **`putchar`** / **`getchar`** in the `.ll` for those paths.
- Without it: legacy I/O uses **`printf`** / **`putchar`** / **`getchar`**.

`malloc` / `free`, `@llvm.sqrt.f64`, and **`@bf2.rx.*`** / **`@bf2.watch.*`** helpers are **`declare`**\ d or **`define`**\ d in the same `.ll` as needed.

## Per-node hints

Each dataclass in `bf2/ast_nodes.py` implements `to_llvm_ir_hint()` for documentation; codegen uses the typed AST directly.

## Pipeline

1. **Lower** `Module`: `type` per `struct`, globals for segments, `alloca` for function-local `seg`, `define` per `fn`.
2. **Watch** (future): private `reactor_*` functions and `call` after stores to watched slots.
3. **I/O**: either **`write`** / **`read`** / **`snprintf`** (stdlib include) or **`printf`** / **`putchar`** / **`getchar`** (legacy).

## SSA alignment

- Reactive firing after `store` matches “observe post-dominating store” in SSA terms.
- `phi` nodes for BF `[` loops match the hint on `LoopBF`.
