import time as t
import numpy as np
from src.model import IsingModel
from src.pythonbenchmark import python_metropolis

def benchmark(L=50, T=2.27, J=1.0, total_steps=1000000):
    lattice = np.random.choice([-1, 1], size = (L, L)). astype(np.int8)
    start = t.perf_counter()
    python_metropolis(lattice, T, J, total_steps)
    end = t.perf_counter()
    print(f"Python implementation took {end - start:.4f} seconds")
    py_time = end - start

    sim = IsingModel(L, T, J)
    start = t.perf_counter()
    sim.run_simulation(total_steps)
    end = t.perf_counter()
    print(f"Cython implementation took {end - start:.4f} seconds")
    cy_time = end - start  
    speedup = py_time / cy_time if cy_time > 0 else float('inf')
    print(f"\nResult: Cython is {speedup:.2f}x faster than Pure Python.")
if __name__ == "__main__":
    benchmark()