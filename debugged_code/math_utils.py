from __future__ import annotations

import math
from typing import Iterable


def is_prime(n: int) -> bool:
    """Return True if n is prime.

    Fixes common logic bugs:
    - Handles n <= 1 correctly (not prime)
    - Treats 2 as prime and filters evens fast
    - Checks divisors only up to sqrt(n)
    - Skips even divisors for performance
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return n == 3

    limit = int(math.isqrt(n))
    # Check 6k ± 1 pattern
    k = 5
    while k <= limit:
        if n % k == 0 or n % (k + 2) == 0:
            return False
        k += 6
    return True


def first_k_primes(k: int) -> list[int]:
    """Generate the first k primes using is_prime.

    Includes input validation that previously could cause logic errors.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    primes: list[int] = []
    x = 2
    while len(primes) < k:
        if is_prime(x):
            primes.append(x)
        x += 1
    return primes
