import pandas as pd
import os
import numpy as np
from src.model import IsingModel
from src.visualization import create_interactive_plot

def main():
    print("--- Starting Ising Model Simulation ---")
    csv_path = 'data/input/parameters.csv'
    params_df = pd.read_csv(csv_path)
    params = params_df.iloc[0].to_dict()
    
    L, T, J, steps = int(params['L']), float(params['T']), float(params['J']), int(params['steps'])
    
    sim = IsingModel(L=L, T=T, J=J)
    history = []
    
    interval = 1000 if steps >= 1000 else steps // 10
    if interval == 0: interval = 1
    num_snapshots = steps // interval

    print(f"Running {steps} steps. Recording {num_snapshots} snapshots...")
    
    for i in range(num_snapshots):
        sim.run_simulation(total_steps=interval)
        history.append(sim.lattice.copy())
    
    history_path = "data/output/history/sim_history.npz"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    np.savez_compressed(history_path, *history)
    
    print(f"History saved to {history_path}. Opening slider...")
    create_interactive_plot(history_path)

if __name__ == "__main__":
    main()