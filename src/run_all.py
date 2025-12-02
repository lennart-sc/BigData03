import os
import numpy as np

from basic_python import generate_matrices_list, matmul_basic
from vectorized_numpy import generate_matrices_np, matmul_numpy
from parallel_numba import matmul_parallel
from utils import time_function, compute_speedup, compute_efficiency, save_results


def run_experiments():
    # Matrix sizes to test
    sizes = [128, 256, 512]  # You can adjust these

    # Pure Python triple-loop will be VERY slow for large n
    # so we won't go crazy with basic for huge sizes.
    max_basic_size = 256

    num_cores = os.cpu_count() or 1
    all_results = []

    for n in sizes:
        print(f"\n========== n = {n} ==========")

        # ----- BASIC (Python lists) -----
        if n <= max_basic_size:
            print("Running basic (pure Python) version...")
            A_list, B_list = generate_matrices_list(n, seed=42)
            t_basic = time_function(matmul_basic, A_list, B_list, repeats=1)
            print(f"  basic time      : {t_basic:.6f} s")
        else:
            t_basic = None
            print("  basic version skipped (too slow for this size).")

        # ----- NUMPY (vectorized) -----
        print("Running NumPy (vectorized) version...")
        A_np, B_np = generate_matrices_np(n, seed=42)
        t_numpy = time_function(matmul_numpy, A_np, B_np, repeats=3)
        print(f"  NumPy time      : {t_numpy:.6f} s")

        # ----- NUMBA (parallel) -----
        print("Running Numba (parallel) version...")
        # Warmup happens inside time_function via warmup=True
        t_parallel = time_function(matmul_parallel, A_np, B_np, repeats=3, warmup=True)
        print(f"  Numba parallel  : {t_parallel:.6f} s")

        # ----- Speedups & efficiency -----
        if t_basic is not None:
            speedup_numpy = compute_speedup(t_basic, t_numpy)
            speedup_parallel = compute_speedup(t_basic, t_parallel)
            efficiency_parallel = compute_efficiency(speedup_parallel, num_cores)

            print(f"  speedup (NumPy vs basic)    : {speedup_numpy:.2f}x")
            print(f"  speedup (parallel vs basic) : {speedup_parallel:.2f}x")
            print(f"  efficiency (parallel)       : {efficiency_parallel:.3f}")
        else:
            speedup_numpy = None
            speedup_parallel = None
            efficiency_parallel = None

        result = {
            "n": n,
            "basic_time": t_basic,
            "numpy_time": t_numpy,
            "parallel_time": t_parallel,
            "speedup_numpy": speedup_numpy,
            "speedup_parallel": speedup_parallel,
            "efficiency_parallel": efficiency_parallel,
            "num_cores": num_cores,
        }
        all_results.append(result)

    # Save results to a JSON file
    out_path = os.path.join(os.path.dirname(__file__), "..", "results.json")
    save_results(out_path, {"results": all_results})
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    run_experiments()

