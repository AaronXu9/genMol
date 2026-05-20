"""Generic typed registry for Hydra-instantiable components."""
from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, type[T]] = {}

    def register(self, key: str):
        def decorator(cls):
            if key in self._items:
                raise KeyError(f"Registry {self.name} already has key {key!r}")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        if key not in self._items:
            raise KeyError(f"Unknown {self.name}: {key!r}. Available: {sorted(self._items)}")
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items
