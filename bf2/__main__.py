"""BF2 CLI entrypoint — run / compile subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bf2.core.errors import BF2Error, BF2RuntimeError, BF2SyntaxError, BF2TypeError, format_error
from bf2.compiler.parser import parse_source
from bf2.compiler.preprocess import Preprocessor, PreprocessError
from bf2.compiler.typechecker import check_module
from bf2.backends.interpreter.engine import Interpreter
from bf2.backends.llvm.emitter import emit_llvm_ir
from bf2.backends.llvm.emit_mem import LLVMGenError
from bf2.cli import parse_compile_args


def _detect_target() -> str:
    """Detect the host target triple for LLVM IR emission.

    Queries clang for its default target triple first, falling back to
    constructing one from platform.machine() with an ``unknown`` vendor.
    """
    import shutil
    import subprocess

    clang = shutil.which("clang")
    if clang:
        try:
            res = subprocess.run(
                [clang, "-dumpmachine"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                triple = res.stdout.strip()
                if triple:
                    return triple
        except (OSError, subprocess.TimeoutExpired):
            pass

    import platform
    arch = platform.machine()
    return f"{arch}-unknown-linux-elf"


def _run_opt(ir: str, level: int, extra_flags: str) -> str:
    """Run LLVM ``opt`` at the given optimization level."""
    import subprocess

    cmd = ["opt", f"-O{level}", "-S"]
    if extra_flags:
        cmd.extend(extra_flags.split())

    res = subprocess.run(cmd, input=ir.encode("utf-8"), capture_output=True)
    if res.returncode == 0:
        return res.stdout.decode("utf-8")
    sys.stderr.write(f"opt failed: {res.stderr.decode('utf-8')}\n")
    return ir


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "linux":
        sys.stderr.write("bf2: error: this compiler and runtime only support Linux.\n")
        return 1

    p = argparse.ArgumentParser(prog="bf2")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- run subcommand (unchanged) ---
    r = sub.add_parser("run", help="run a .bf2 file in the interpreter")
    r.add_argument("path", help="source file")
    r.add_argument("args", nargs=argparse.REMAINDER, help="arguments to pass to the program")

    # --- compile subcommand (GCC-style flags) ---
    c = sub.add_parser(
        "compile", help="emit LLVM IR",
        # Disable prefix matching so -O3 doesn't get swallowed by argparse
        allow_abbrev=False,
    )
    c.add_argument("compile_argv", nargs=argparse.REMAINDER,
                    help="source file and flags (-O1, -fliro, -Wall, etc.)")

    args = p.parse_args(argv)

    # --- run ---
    if args.cmd == "run":
        path = Path(args.path).resolve()
        try:
            prep = Preprocessor(path)
            src, meta = prep.process_file(path)
            mod = parse_source(src, use_linux_stdlib=meta.use_linux_stdlib)
            check_module(mod)
            interp = Interpreter(mod)
            out = interp.run([args.path] + args.args)
            sys.stdout.write(out)
            return 0
        except PreprocessError as e:
            sys.stderr.write(f"PreprocessError: {e}\n")
            return 2
        except BF2SyntaxError as e:
            sys.stderr.write(format_error(e, src, "BF2SyntaxError") + "\n")
            return 2
        except BF2TypeError as e:
            sys.stderr.write(format_error(e, src, "BF2TypeError") + "\n")
            return 3
        except BF2RuntimeError as e:
            sys.stderr.write(format_error(e, src, "BF2RuntimeError") + "\n")
            return 4
        except BF2Error as e:
            sys.stderr.write(format_error(e, src, "BF2Error") + "\n")
            return 5
        except Exception as e:
            sys.stderr.write(f"Internal Error: {e}\n")
            import traceback
            traceback.print_exc()
            return 1

    # --- compile ---
    if args.cmd == "compile":
        opts = parse_compile_args(args.compile_argv)
        if not opts.path:
            sys.stderr.write("bf2 compile: error: no source file specified\n")
            return 1

        path = Path(opts.path).resolve()
        src = ""

        try:
            prep = Preprocessor(path)
            src, meta = prep.process_file(path)

            # Parse + typecheck
            from bf2.compiler.diagnostics import DiagnosticCollector
            mod = parse_source(src, use_linux_stdlib=meta.use_linux_stdlib)
            dc = DiagnosticCollector(enabled=set(opts.warnings))
            check_module(mod, diag=dc)

            # Print diagnostics if any
            if dc.diagnostics:
                sys.stderr.write(dc.format_all(src) + "\n")

            # Emit LLVM IR
            target = opts.target or _detect_target()
            module = emit_llvm_ir(mod, target)

            # LIRO passes (optional, off by default)
            if opts.liro_enabled:
                from bf2.liro.runner import resolve_liro_spec, run_liros
                pass_names = resolve_liro_spec(opts.liro_spec)
                ir = run_liros(str(module), pass_names)
            else:
                ir = str(module)

            # LLVM opt (optional, -O1 through -O3)
            if opts.opt_level > 0:
                ir = _run_opt(ir, opts.opt_level, opts.additional_optflags)

            # Output
            if opts.output:
                Path(opts.output).write_text(ir, encoding="utf-8")
            else:
                sys.stdout.write(ir)
            return 0

        except PreprocessError as e:
            sys.stderr.write(f"PreprocessError: {e}\n")
            return 2
        except BF2SyntaxError as e:
            sys.stderr.write(format_error(e, src, "BF2SyntaxError") + "\n")
            return 2
        except BF2TypeError as e:
            sys.stderr.write(format_error(e, src, "BF2TypeError") + "\n")
            return 3
        except LLVMGenError as e:
            sys.stderr.write(f"LLVMGenError: {e}\n")
            return 4
        except BF2Error as e:
            sys.stderr.write(format_error(e, src, "BF2Error") + "\n")
            return 5
        except Exception as e:
            sys.stderr.write(f"Internal Error: {e}\n")
            import traceback
            traceback.print_exc()
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
