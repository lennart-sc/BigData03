import json
import os
from time import perf_counter
from typing import Callable, Any


def time_function(
    fn: Callable[..., Any],
    *args,
    repeats: int = 3,
    warmup: bool = False,
) -> float:
    """
    Time a function; return the best (minimum) time over 'repeats' runs.

    If warmup=True, calls fn once before timing (useful for Numba).
    """
    if warmup:
        fn(*args)

    best = float("inf")
    for _ in range(repeats):
        start = perf_counter()
        fn(*args)
        end = perf_counter()
        elapsed = end - start
        if elapsed < best:
            best = elapsed
    return best


def compute_speedup(t_base: float, t_other: float) -> float:
    """
    Compute speedup = t_base / t_other.
    """
    if t_other <= 0:
        return float("inf")
    return t_base / t_other


def compute_efficiency(speedup: float, num_threads: int) -> float:
    """
    Efficiency = speedup / num_threads.
    """
    if num_threads <= 0:
        return 0.0
    return speedup / num_threads


def save_results(path: str, data: dict):
    """
    Save results dictionary to JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
