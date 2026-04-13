import numpy as np
import pandas as pd 
import time as t
import os
from src.model import IsingModel as SerialModel
from src.parallel_model import IsingModel as ParallelModel 

def pareto_analysis(size=1, alpha=1.5):
    return (1.0 - np.random.random(size)) ** (-1.0 / (alpha - 1.0))

def run_stress_test(num_sims=100):
    results = []
    base_flips = 100000 
    T = 2.27
    J = 1.0
    
    print(f"Executing {num_sims} sims comparing Serial vs Parallel/1D")

    for i in range(num_sims):
        raw_sample = pareto_analysis(1, 1.5)[0]
        L = 512 # min(int(10 + (raw_sample * 20)) , 512) 

        total_steps = min(int(base_flips * raw_sample), 10000)
        
        # 1. SERIAL CYTHON 
        sim_s = SerialModel(L=L, T=T, J=J) 
        start_s = t.perf_counter()
        sim_s.run_simulation(total_steps=total_steps)
        serial_time = t.perf_counter() - start_s

        # 2. PARALLEL CYTHON 
        sweeps = max(1, total_steps // (L * L))
        actual_parallel_flips = sweeps * (L * L)
        
        sim_p = ParallelModel(L=L, T=T, J=J)
        start_p = t.perf_counter()
        sim_p.run_simulation(sweeps=sweeps)
        parallel_time = t.perf_counter() - start_p

        normalized_speedup = (serial_time / parallel_time) * (actual_parallel_flips / total_steps) if parallel_time > 0 else 0

        results.append({
            'L': L,
            'total_steps': total_steps,
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': normalized_speedup
        })

        if i % 10 == 0:
            print(f"Sim {i}: L={L}, Speedup={normalized_speedup:.2f}x")

    df = pd.DataFrame(results)
    
    
    os.makedirs(os.path.dirname('data/output/stress_test_parallel.csv'), exist_ok=True)
    df.to_csv('data/output/stress_test_parallel.csv', mode='a', index=False, header=True)
    
    full_df = pd.read_csv('data/output/stress_test_parallel.csv')
    
    print("    Cumulative Stress Test Stats    ")
    print("-"*40)
    print(f"Total Samples in CSV: {len(full_df)}")
    print(f"Mean Speedup:         {full_df['speedup'].mean():.2f}x")
    print(f"Median Speedup:       {full_df['speedup'].median():.2f}x")
    print(f"Max Speedup:          {full_df['speedup'].max():.2f}x")
    print(f"Win Rate (Parallel):  {(full_df['speedup'] > 1.0).mean()*100:.1f}%")

if __name__ == "__main__":
    run_stress_test(100)