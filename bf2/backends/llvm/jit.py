from __future__ import annotations

import ctypes
from typing import List

import llvmlite.binding as llvm


def run_jit(ir_str: str, args: List[str]) -> int:
    """Execute the provided LLVM IR using JIT.

    Args:
        ir_str: The LLVM assembly string.
        args: Command line arguments to pass to main (including program name).

    Returns:
        The exit code from main.
    """
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # Create a target machine
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()

    # Parse IR and create a module
    module = llvm.parse_assembly(ir_str)
    module.verify()

    # Add the module to an execution engine
    # Use MCJIT as it's the standard for llvmlite
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    engine.add_module(module)
    engine.finalize_object()
    engine.run_static_constructors()

    # Get function pointer for main
    func_ptr = engine.get_function_address("main")
    if not func_ptr:
        raise RuntimeError("No main() function found in LLVM IR")

    # Define main's signature: int main(int argc, char** argv)
    main_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
    main_func = main_type(func_ptr)

    # Prepare argc and argv
    argc = len(args)
    argv_type = ctypes.c_char_p * argc
    argv = argv_type(*[arg.encode("utf-8") for arg in args])

    # Execute
    try:
        return main_func(argc, argv)
    finally:
        engine.run_static_destructors()
