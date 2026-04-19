from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bf2.core.errors import BF2Error, BF2RuntimeError, BF2SyntaxError, BF2TypeError, format_error
from bf2.compiler.parser import parse_source
from bf2.compiler.preprocess import Preprocessor, PreprocessError
from bf2.compiler.typechecker import check_module
from bf2.backends.interpreter.engine import Interpreter
from bf2.backends.llvm.emitter import LLVMEmitterVisitor, LLVMGenError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bf2")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    r = sub.add_parser("run", help="run a .bf2 file in the interpreter")
    r.add_argument("path", help="source file")
    r.add_argument("args", nargs=argparse.REMAINDER, help="arguments to pass to the program")
    
    c = sub.add_parser("compile", help="emit LLVM IR")
    c.add_argument("path", help="source file")
    c.add_argument("-o", "--output", metavar="FILE", help="write .ll here")
    c.add_argument("-O", "--optimize", action="store_true", help="optimize with opt -O3")
    c.add_argument("--target", help="target triple")
    
    args = p.parse_args(argv)
    path = Path(args.path).resolve()
    
    try:
        prep = Preprocessor(path)
        src, meta = prep.process_file(path)
        
        mod = parse_source(src, use_linux_stdlib=meta.use_linux_stdlib)
        check_module(mod)
        
        if args.cmd == "run":
            interp = Interpreter(mod)
            out = interp.run([args.path] + args.args)
            sys.stdout.write(out)
            return 0
            
        if args.cmd == "compile":
            import platform
            target = args.target
            if not target:
                machine = platform.machine()
                target = f"{machine}-pc-linux-gnu" if machine not in ("aarch64", "arm64") else "aarch64-linux-gnu"
            
            emitter = LLVMEmitterVisitor(mod, target)
            ir = emitter.emit()
            
            if args.optimize:
                import subprocess
                res = subprocess.run(["opt", "-O3", "-S"], input=ir.encode("utf-8"), capture_output=True)
                if res.returncode == 0:
                    ir = res.stdout.decode("utf-8")
                else:
                    sys.stderr.write(f"opt failed: {res.stderr.decode('utf-8')}\n")
            
            if args.output:
                Path(args.output).write_text(ir, encoding="utf-8")
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
    except BF2RuntimeError as e:
        sys.stderr.write(format_error(e, src, "BF2RuntimeError") + "\n")
        return 4
    except LLVMGenError as e:
        sys.stderr.write(f"LLVMGenError: {e}\n")
        return 4
    except BF2Error as e:
        sys.stderr.write(format_error(e, src, "BF2Error") + "\n")
        return 5
    except Exception as e:
        sys.stderr.write(f"Internal Error: {e}\n")
        import traceback; traceback.print_exc()
        return 1
        
    return 1


if __name__ == "__main__":
    sys.exit(main())
