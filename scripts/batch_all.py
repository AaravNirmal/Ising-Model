import numpy as np
import pandas as pd 
import time as t
import os
from src.pythonbenchmark import python_metropolis as PyModel
from src.model import IsingModel as SerialModel
from src.parallel_model import IsingModel as ParallelModel 

def pareto_analysis(size=1, alpha=1.5):
    return (1.0 - np.random.random(size)) ** (-1.0 / (alpha - 1.0))

def run_stress_test(num_sims):
    base_flips = 100000 
    T = 2.269
    J = 1.0
    output_path = 'data/output/stress_test_all_new.csv'
    
    print(f"Executing {num_sims} sims: Python vs Serial Cython vs Parallel/1D", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i in range(num_sims):
        t_variance = np.random.normal(0, 0.2)
        T += min(2.0, max(-2.0, t_variance))
        raw_sample = pareto_analysis(1, 1.5)[0]
        L = int(2**np.random.uniform(5, 10))
        total_steps = min(int(base_flips * raw_sample), 1e9)
        
        # 1. VANILLA PYTHON (The Baseline)
        py_steps = int(total_steps if L <= 128 else 10000)
        lattice_py = np.random.choice([1, -1], size=(L, L))
        
        start_py = t.perf_counter()
        PyModel(lattice_py, T, J, py_steps)
        python_time = (t.perf_counter() - start_py) * (total_steps / py_steps)

        # 2. SERIAL CYTHON 
        sim_s = SerialModel(L=L, T=T, J=J) 
        start_s = t.perf_counter()
        sim_s.run_simulation(total_steps=total_steps)
        serial_time = t.perf_counter() - start_s

        # 3. PARALLEL CYTHON (1D Optimized)
        sweeps = max(1, total_steps // (L * L))
        actual_parallel_flips = sweeps * (L * L)
        
        sim_p = ParallelModel(L=L, T=T, J=J)
        start_p = t.perf_counter()
        sim_p.run_simulation(sweeps=sweeps)
        parallel_time = t.perf_counter() - start_p

        cython_boost = python_time / serial_time if serial_time > 0 else 0
        parallel_boost = (serial_time / parallel_time) * (actual_parallel_flips / total_steps) if parallel_time > 0 else 0

        # Construct single dictionary row directly for immediate disk dump
        single_result = {
            'L': L,
            'total_steps': total_steps,
            'python_time': python_time,
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'cython_boost': cython_boost,
            'parallel_boost': parallel_boost,
            'total_speedup': python_time / (parallel_time * (total_steps / actual_parallel_flips))
        }

        # Write to disk instantly so you see performance updates line-by-line
        df_single_row = pd.DataFrame([single_result])
        header_needed = not os.path.exists(output_path)
        df_single_row.to_csv(output_path, mode='a', index=False, header=header_needed)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_sims} simulations... (Last L: {L})", flush=True)

    print(f"\nBenchmark complete. Data appended to {output_path}", flush=True)

if __name__ == "__main__":
    run_stress_test(num_sims=10)