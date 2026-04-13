import time as t
import numpy as np
from src.model import IsingModel as CythonIsingModel
from src.parallel_model import IsingModel as ParallelCythonIsingModel

def benchmark(L=256, T=2.27, J=1.0, total_steps=50000000):
    # 1. Cython Serial
    sim_cy = CythonIsingModel(L, T, J)
    start = t.perf_counter()
    sim_cy.run_simulation(total_steps)
    cy_time = t.perf_counter() - start
    
    # 2. Parallel Cython (Normalized Workload)
    sweeps = max(1, total_steps // (L * L)) 
    actual_parallel_steps = sweeps * (L * L)
    
    sim_p = ParallelCythonIsingModel(L, T, J)
    start = t.perf_counter()
    sim_p.run_simulation(sweeps)
    p_cy_time = t.perf_counter() - start

    normalized_p_time = p_cy_time * (total_steps / actual_parallel_steps)

    print(f" Ising Model Cython Comparison (L={L}, Steps={total_steps})")
    print("-"*55)
    print(f"{'Implementation':<25} | {'Time (s)':<10} | {'Rel. Speed'}")
    print(f"{'Cython (Serial)':<25} | {cy_time:<10.4f} | 1.00x (Base)")
    print(f"{'Cython (Parallel)':<25} | {normalized_p_time:<10.4f} | {cy_time/normalized_p_time:.2f}x")
    
    print(f"Parallel throughput: {total_steps / normalized_p_time / 1e6:.2f} Million flips/sec")

if __name__ == "__main__":
    benchmark(L=2048, total_steps=1e8)