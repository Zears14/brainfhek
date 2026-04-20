"""GCC/Clang-style CLI flag parsing for the ``compile`` subcommand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompileOptions:
    """Parsed representation of all ``bf2 compile`` flags."""

    path: str = ""

    # Output
    output: Optional[str] = None

    # Optimization level (0 = none, 1-3 = opt -O1/-O2/-O3)
    opt_level: int = 0

    # LIRO
    liro_enabled: bool = False
    liro_spec: str = ""       # raw value after -fliro=

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Target
    target: Optional[str] = None

    # Additional opt flags passed verbatim
    additional_optflags: str = ""


def parse_compile_args(argv: List[str]) -> CompileOptions:
    """Parse GCC/Clang-style flags from the compile subcommand arguments.

    Recognized flags:
        <path>                       Source file (positional, required)
        -o FILE                      Output file
        -O1, -O2, -O3               Optimization level for ``opt``
        -fliro                       Enable all default LIROs
        -fliro=SPEC                  Enable specific LIROs (N, name, or name,name)
        -fno-liro                    Explicitly disable LIROs
        -Wall                        Enable all warnings
        -Wub                         Warn on undefined behavior
        -Wunreachable                Warn on unreachable code
        -Wshadow                     Warn on shadowed variables
        -Wunused                     Warn on unused variables
        -Wno-<name>                  Disable a specific warning
        --target=TRIPLE              Override target triple
        --additional-optflags="..."  Append raw flags to ``opt``
    """
    opts = CompileOptions()
    i = 0

    while i < len(argv):
        arg = argv[i]

        # -o FILE
        if arg == "-o":
            i += 1
            if i < len(argv):
                opts.output = argv[i]
            i += 1
            continue

        # -O1, -O2, -O3
        if arg in ("-O1", "-O2", "-O3"):
            opts.opt_level = int(arg[2])
            i += 1
            continue

        # -fliro or -fliro=SPEC
        if arg == "-fliro":
            opts.liro_enabled = True
            opts.liro_spec = ""
            i += 1
            continue
        if arg.startswith("-fliro="):
            opts.liro_enabled = True
            opts.liro_spec = arg[7:]
            i += 1
            continue

        # -fno-liro
        if arg == "-fno-liro":
            opts.liro_enabled = False
            i += 1
            continue

        # Warnings: -Wall, -Wub, -Wunreachable, -Wshadow, -Wunused, -Wno-*
        if arg == "-Wall":
            opts.warnings = ["ub", "unreachable", "shadow", "unused"]
            i += 1
            continue
        if arg.startswith("-Wno-"):
            name = arg[5:]
            if name in opts.warnings:
                opts.warnings.remove(name)
            i += 1
            continue
        if arg.startswith("-W") and len(arg) > 2:
            name = arg[2:]
            if name not in opts.warnings:
                opts.warnings.append(name)
            i += 1
            continue

        # --target=TRIPLE
        if arg.startswith("--target="):
            opts.target = arg[9:]
            i += 1
            continue
        if arg == "--target":
            i += 1
            if i < len(argv):
                opts.target = argv[i]
            i += 1
            continue

        # --additional-optflags="..."
        if arg.startswith("--additional-optflags="):
            opts.additional_optflags = arg[22:].strip('"').strip("'")
            i += 1
            continue
        if arg == "--additional-optflags":
            i += 1
            if i < len(argv):
                opts.additional_optflags = argv[i]
            i += 1
            continue

        # Positional: source file path
        if not arg.startswith("-"):
            opts.path = arg
            i += 1
            continue

        # Unknown flag — skip (could raise error later)
        i += 1

    return opts
