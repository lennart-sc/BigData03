import numpy as np
from time import perf_counter


def generate_matrices_np(n: int, seed: int | None = None):
    """
    Generate two n x n NumPy matrices of random floats in [0, 1).
    """
    if seed is not None:
        np.random.seed(seed)

    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A, B


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Vectorized matrix multiplication using NumPy (uses optimized BLAS/SIMD).
    """
    return A @ B  # or np.dot(A, B)


if __name__ == "__main__":
    n = 1024
    A, B = generate_matrices_np(n)

    start = perf_counter()
    C = matmul_numpy(A, B)
    end = perf_counter()

    print(f"[vectorized_numpy] n={n}, time={end - start:.6f} s")

