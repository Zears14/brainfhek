"""Base class and registration decorator for LIRO passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class LIROPass(ABC):
    """Abstract base for all LIRO (LLVM IR Optimization) passes.

    A LIRO operates on raw IR text lines and returns transformed lines.
    Each pass **must** be independent — it cannot assume any other LIRO
    has run before or after it.
    """

    #: Unique identifier used in ``-fliro=name`` and the registry.
    name: str = ""

    #: One-line description shown in ``--help-liro``.
    description: str = ""

    @abstractmethod
    def run(self, lines: List[str]) -> List[str]:
        """Transform IR lines and return the (possibly modified) list."""
        ...


def register_liro(cls: type[LIROPass]) -> type[LIROPass]:
    """Class decorator that registers a LIRO pass in the global registry."""
    from bf2.liro import LIRO_REGISTRY
    if not cls.name:
        raise ValueError(f"LIRO pass {cls.__name__} must define a 'name' attribute")
    LIRO_REGISTRY[cls.name] = cls
    return cls
