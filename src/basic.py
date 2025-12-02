
import random
from time import perf_counter


def generate_matrices_list(n: int, seed: int | None = None):
    """
    Generate two n x n matrices as Python lists of lists.
    """
    if seed is not None:
        random.seed(seed)

    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    return A, B


def matmul_basic(A, B):
    """
    Basic (sequential) triple-loop matrix multiplication.

    A: list of lists, shape (n, m)
    B: list of lists, shape (m, p)

    Returns C = A * B as list of lists, shape (n, p).
    """
    n = len(A)
    m = len(A[0])
    m2 = len(B)
    p = len(B[0])

    assert m == m2, "Inner dimensions must match"

    C = [[0.0] * p for _ in range(n)]

    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                # Correct formula: B[k][j], not B[k][k]
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


if __name__ == "__main__":
    # Demo: this will be slow if n is large!
    n = 128
    A, B = generate_matrices_list(n)

    start = perf_counter()
    C = matmul_basic(A, B)
    end = perf_counter()

    print(f"[basic_python] n={n}, time={end - start:.6f} s")
