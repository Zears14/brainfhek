from __future__ import annotations

import argparse
import sys

from bf2.errors import BF2Error, BF2RuntimeError, BF2SyntaxError, BF2TypeError, format_error
from bf2.interpreter import Interpreter, is_classic_bf, run_bf_classic
from bf2.llvm_emit import LLVMGenError, emit_llvm_ir
from bf2.parser import parse_source
from bf2.preprocess import PreprocessError, preprocess_path
from bf2.typechecker import check_module


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bf2")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a .bf2 or classic .bf file")
    r.add_argument("path", help="source file")
    c = sub.add_parser("compile", help='emit standalone LLVM IR (with #include "stdlib": write/read/snprintf; else printf/putchar)')
    c.add_argument("path", help="source file")
    c.add_argument("-o", "--output", metavar="FILE", help="write .ll here (default: stdout)")
    c.add_argument("-O", "--optimize", action="store_true", help="run opt -O3 to generate more efficient IR")
    c.add_argument("--target", help="target triple (default: host, must be linux-gnu)")
    args = p.parse_args(argv)
    if args.cmd == "compile":
        from pathlib import Path

        path = Path(args.path)
        try:
            import platform
            target = args.target
            if not target:
                machine = platform.machine()
                target = f"{machine}-pc-linux-gnu" if machine not in ("aarch64", "arm64") else "aarch64-linux-gnu"
            if "linux" not in target:
                sys.stderr.write("error: target must be a linux target triple\n")
                return 1

            src, meta = preprocess_path(path.resolve())
            mod = parse_source(src, use_linux_stdlib=meta.use_linux_stdlib)
            check_module(mod)
            ir = emit_llvm_ir(mod, target)
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
            
        if getattr(args, "optimize", False):
            import subprocess
            p = subprocess.run(["opt", "-O3", "-S"], input=ir.encode("utf-8"), capture_output=True)
            if p.returncode == 0:
                ir = p.stdout.decode("utf-8")
            else:
                sys.stderr.write("opt failed:\n" + p.stderr.decode("utf-8") + "\n")
                return 5
        if args.output:
            with open(args.output, "w", encoding="utf-8") as out:
                out.write(ir)
        else:
            sys.stdout.write(ir)
        return 0
    if args.cmd == "run":
        from pathlib import Path

        path = Path(args.path)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        if is_classic_bf(src):
            sys.stdout.write(run_bf_classic(src))
            return 0
        try:
            src, meta = preprocess_path(path.resolve())
            mod = parse_source(src, use_linux_stdlib=meta.use_linux_stdlib)
            check_module(mod)
            ip = Interpreter(mod)
            ip.run()
            sys.stdout.write("".join(ip.out))
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
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
