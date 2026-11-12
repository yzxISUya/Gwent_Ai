"""
Deterministic 32-bit PRNG (mulberry32) bit-for-bit identical to JavaScript's
Math.imul and unsigned 32-bit shifts (>>> 0).
"""
from __future__ import annotations
from typing import List, TypeVar

T = TypeVar("T")


def imul32(a: int, b: int) -> int:
    """Equivalent to (Math.imul(a, b) >>> 0) in JavaScript."""
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


def hash_seed(seed: int | str | None) -> int:
    """Equivalent to hashSeed(seed) in JavaScript."""
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed & 0xFFFFFFFF
    s = "gwent" if seed is None else str(seed)
    h = 2166136261 & 0xFFFFFFFF
    for ch in s:
        h ^= ord(ch)
        h = imul32(h, 16777619)
    return h & 0xFFFFFFFF


class RngState:
    __slots__ = ("state",)

    def __init__(self, seed: int | str = 1):
        self.state: int = hash_seed(seed)

    def clone(self) -> "RngState":
        r = RngState.__new__(RngState)
        r.state = self.state
        return r

    def next_float(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = imul32(t ^ (t >> 15), t | 1)
        t = (t ^ (t + imul32(t ^ (t >> 7), t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    def rand_int(self, n: int) -> int:
        if n <= 0:
            return 0
        return int(self.next_float() * n)

    def shuffle(self, arr: List[T]) -> List[T]:
        out = list(arr)
        for i in range(len(out) - 1, 0, -1):
            j = self.rand_int(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def pick_random(self, arr: List[T]) -> T | None:
        if not arr:
            return None
        return arr[self.rand_int(len(arr))]
