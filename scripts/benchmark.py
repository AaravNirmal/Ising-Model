import time as t
import numpy as np
from src.model import IsingModel as CythonIsingModel
from src.parallel_model import IsingModel as ParallelCythonIsingModel
from src.pythonbenchmark import python_metropolis

def benchmark(L=500, T=2.27, J=1.0, total_steps=1000000):
    lattice = np.random.choice([-1, 1], size = (L, L)). astype(np.int8)
    start = t.perf_counter()
    sim_p = python_metropolis(lattice, T, J, total_steps)
    end = t.perf_counter()
    print(f"Python implementation took {end - start:.4f} seconds")
    py_time = end - start

    sim_cy = CythonIsingModel(L, T, J)
    start = t.perf_counter()
    sim_cy.run_simulation(total_steps)
    end = t.perf_counter()
    print(f"Cython implementation took {end - start:.4f} seconds")
    cy_time = end - start  
    speedup = py_time / cy_time if cy_time > 0 else float('inf')
    
    sweeps = max(1, total_steps // (L * L)) 
    sim_p = ParallelCythonIsingModel(L, T, J)
    start = t.perf_counter()
    sim_p.run_simulation(sweeps)
    p_cy_time = t.perf_counter() - start

    speedup = py_time / p_cy_time if p_cy_time > 0 else float('inf')
    print("\n" + "="*50)
    print(f" Ising Model Benchmark (L={L}, Steps={total_steps})")
    print("="*50)
    print(f"{'Implementation':<25} | {'Time (s)':<10} | {'Speedup':<10}")
    print("-" * 50)
    print(f"{'Pure Python':<25} | {py_time:<10.4f} | {'1.00x':<10}")
    print(f"{'Cython (Serial)':<25} | {cy_time:<10.4f} | {py_time/cy_time:<10.2f}x")
    print(f"{'Cython (Parallel)':<25} | {p_cy_time:<10.4f} | {py_time/p_cy_time:<10.2f}x")
    print("-" * 50)
    
    if p_cy_time > 0:
        scaling = cy_time / p_cy_time
        print(f"Parallel is {scaling:.2f}x faster than Serial Cython.")
    print("="*50)

if __name__ == "__main__":
    benchmark()