from __future__ import annotations

from typing import Iterable


def concat_strings(parts: Iterable[str], separator: str = "") -> str:
    """Concatenate strings efficiently.

    Fixes performance bug: previously used repeated `+` concatenation in a loop,
    leading to quadratic complexity. This implementation uses `str.join`, which
    is linear and memory-efficient.
    """
    if parts is None:
        return ""
    # Convert to list once to avoid consuming an iterator twice unexpectedly
    items = list(parts)
    # Normalize None entries defensively
    items = [p if p is not None else "" for p in items]
    return separator.join(items)
