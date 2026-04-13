import numpy as np
import pandas as pd 
import time as t
from src.model import IsingModel
from src.pythonbenchmark import python_metropolis   

def pareto_analysis(size = 1, alpha = 1.5):
    return (1.0 - np.random.random(size)) ** (-1.0 / (alpha - 1.0))

def run_stress_test(num_sims = 100):
    results = []

    lambda_steps = 1000
    T = 2.27
    J = 1.0
    print(f"Executing {num_sims} sims with pareto-distributed steps")

    for i in range (num_sims):
        raw_sample = pareto_analysis(1, 1.5)[0]

        L = int(10+(raw_sample * 5))
        L = min(L, 100)

        steps = int(lambda_steps * raw_sample)
        
        
        sim = IsingModel(L=L, T=T, J=J) 
        startC = t.perf_counter()
        sim.run_simulation(total_steps=steps)
        endC = t.perf_counter()
        cy_time = endC - startC

        lattice = np.random.choice([-1, 1], size = (L, L)). astype(np.int8)
        startP = t.perf_counter()
        python_metropolis(lattice, T=T, J=J, total_steps=steps)
        endP = t.perf_counter()
        py_time = endP - startP
        speedup = py_time / cy_time if cy_time > 0 else 0

        results.append({
            'L': L,
            'steps': steps,
            'cy_time': cy_time,
            'py_time': py_time,
            'speedup': py_time / cy_time if not np.isnan(py_time) else np.nan
        })

    df = pd.DataFrame(results)
    output_path = 'data/output/stress_test_results.csv'
    df.to_csv(output_path, mode='a', index=False)
    print("\n    Stress Test Complete    ")
    print(f"Avg Python Time: {df['py_time'].mean():.4f}s")
    print(f"Avg Cython Time: {df['cy_time'].mean():.4f}s")
    print(f"Max Speedup Recorded: {df['speedup'].max():.2f}x")


if __name__ == "__main__":
    run_stress_test()