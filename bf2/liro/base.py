"""Base class and registration decorator for LIRO passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llvmlite import binding


class LIROPass(ABC):
    """Abstract base for all LIRO (LLVM IR Optimization) passes.

    A LIRO operates on llvmlite binding.ModuleRef and returns a (possibly modified) module.
    Each pass **must** be independent — it cannot assume any other LIRO
    has run before or after it.
    """

    #: Unique identifier used in ``-fliro=name`` and the registry.
    name: str = ""

    #: One-line description shown in ``--help-liro``.
    description: str = ""

    @abstractmethod
    def run(self, module: binding.ModuleRef) -> binding.ModuleRef:
        """Transform ModuleRef and return the (possibly modified) module."""
        ...


def register_liro(cls: type[LIROPass]) -> type[LIROPass]:
    """Class decorator that registers a LIRO pass in the global registry."""
    from bf2.liro import LIRO_REGISTRY
    if not cls.name:
        raise ValueError(f"LIRO pass {cls.__name__} must define a 'name' attribute")
    LIRO_REGISTRY[cls.name] = cls
    return cls