"""LIRO — LLVM IR Optimization pass framework.

Each LIRO pass operates on raw IR text lines and is independent of all
other passes.  Use ``-fliro`` to run the default set, ``-fliro=name``
to cherry-pick, or ``-fliro=a,b`` to compose.
"""

from __future__ import annotations

from typing import Dict, List, Type

# Populated by the @register_liro decorator in each pass module
LIRO_REGISTRY: Dict[str, Type] = {}

# Canonical execution order when ``-fliro`` is passed with no arguments
DEFAULT_LIRO_ORDER: List[str] = [
    "static_watch_fold",
    "dead_branch_elim",
    "loop_metadata",
]

# Force-import pass modules so they self-register via the decorator
from bf2.liro import static_watch_fold as _swf  # noqa: F401, E402
from bf2.liro import dead_branch_elim as _dbe   # noqa: F401, E402
from bf2.liro import loop_metadata as _lm       # noqa: F401, E402
