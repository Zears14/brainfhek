from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, List


class ReactorEngine:
    def __init__(self) -> None:
        self.reactors: DefaultDict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._depth = 0

    def register(self, addr: str, fn: Callable[[Any], None]) -> None:
        self.reactors[addr].append(fn)

    def _in_reactor(self) -> bool:
        return self._depth > 0

    def fire(self, addr: str, new_val: Any) -> None:
        if self._depth > 8: # recursion limit
            return
        self._depth += 1
        try:
            for fn in self.reactors.get(addr, []):
                fn(new_val)
        finally:
            self._depth -= 1
