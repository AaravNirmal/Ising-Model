import pandas as pd
import numpy as np
import os

def main():
    path = 'data/output/stress_test_all_new.csv'
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    print("Cython Speedup:" )
    print(df['cython_boost'].describe())
    print("-" * 40)
    print("Parallel Speedup:" )
    print(df['parallel_boost'].describe())
    print("-" * 40)
    print("Total Speedup:" )
    print(df['total_speedup'].describe())
    print("-" * 40)
    print("Python Time:" )
    print(df['python_time'].describe())
    print("-" * 40)
    print("Serial Time:" )
    print(df['serial_time'].describe())
    print("-" * 40)
    print("Parallel Time:" )
    print(df['parallel_time'].describe())
    print("-" * 40)
    print("Python Throughput (Spin-Flips/s):" )
    print((df['total_steps']/df['python_time']).mean())
    print("-" * 40)
    print("Serial Throughput (Spin-Flips/s):" )
    print((df['total_steps']/df['serial_time']).mean())
    print("-" * 40)
    print("Parallel Cython Throughput (Spin-Flips/s):" )
    print((df['total_steps']/df['parallel_time']).mean())
    print("-" * 40)
    print("Average Lattice Size (L):" )
    print(df['L'].mean())
    print("Average Total Steps:" )
    print(df['total_steps'].mean())

    
    


if __name__ == "__main__":
    main()