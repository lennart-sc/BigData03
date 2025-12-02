import numpy as np
from time import perf_counter
from numba import njit, prange


@njit(parallel=True)
def _matmul_parallel_kernel(A, B):
    """
    Numba-parallelized triple-loop matrix multiplication.

    A, B are NumPy arrays.
    """
    n, m = A.shape
    m2, p = B.shape

    # Safety check (Numba-compatible)
    if m != m2:
        raise ValueError("Inner dimensions must match")

    C = np.zeros((n, p))

    # Parallel over rows
    for i in prange(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def matmul_parallel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Wrapper around the Numba-compiled kernel.
    """
    return _matmul_parallel_kernel(A, B)


if __name__ == "__main__":
    n = 1024
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Warm-up (trigger compilation – this run is slower)
    _ = matmul_parallel(A, B)

    start = perf_counter()
    C = matmul_parallel(A, B)
    end = perf_counter()

    print(f"[parallel_numba] n={n}, time={end - start:.6f} s")

