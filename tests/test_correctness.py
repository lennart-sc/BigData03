import os
import sys
import numpy as np

# Make sure 'src' is on the Python path when running this directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from basic_python import matmul_basic
from parallel_numba import matmul_parallel
from vectorized_numpy import matmul_numpy


def test_implementations_agree():
    n = 32

    np.random.seed(0)
    A_np = np.random.rand(n, n)
    B_np = np.random.rand(n, n)

    A_list = A_np.tolist()
    B_list = B_np.tolist()

    C_basic = np.array(matmul_basic(A_list, B_list))
    C_numpy = matmul_numpy(A_np, B_np)
    C_parallel = matmul_parallel(A_np, B_np)

    assert np.allclose(C_basic, C_numpy, atol=1e-6)
    assert np.allclose(C_basic, C_parallel, atol=1e-6)


if __name__ == "__main__":
    test_implementations_agree()
    print("All implementations produce the same result (within tolerance).")
