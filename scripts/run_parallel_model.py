import pandas as pd
import os
import numpy as np
from src.parallel_model import IsingModel
from src.visualization import create_interactive_plot

def main():
    print("   Starting Ising Model Simulation   ")
    csv_path = 'data/input/parameters.csv'
    params_df = pd.read_csv(csv_path)
    params = params_df.iloc[0].to_dict()
    
    L, T, J, steps = int(params['L']), float(params['T']), float(params['J']), int(params['steps'])
    
    sim = IsingModel(L=L, T=T, J=J)
    history = []

    total_sweeps = int(steps // (L * L))
    total_sweeps = max(50, total_sweeps)

    num_snapshots = min(200, total_sweeps) 


    print(f"Running {steps} steps ({total_sweeps} sweeps). Recording {num_snapshots} snapshots...")
    for i in range(int(num_snapshots)):
        sim.run_simulation(sweeps= min (1, total_sweeps / num_snapshots))
        history.append(sim.lattice.reshape(L,L).copy())

    history_path = "data/output/history/sim_history.npz"

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    np.savez_compressed(history_path, *history)
    print(f"History saved to {history_path}")
    create_interactive_plot(history_path)


if __name__ == "__main__":
    main()