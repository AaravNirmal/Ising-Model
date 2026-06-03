import numpy as np
import sys

try:
    from .lang import ising_core_p
    cython_metropolis_parallel = ising_core_p.cython_metropolis
    print("src/model.py: Parallel Cython engine loaded successfully.")
except ImportError as e:
    cython_metropolis_parallel = None
    print(f"src/model.py: Could not load Parallel Cython engine. Error: {e}")

class IsingModel:
    def __init__(self, L, T, J=1.0):
        self.L = L
        self.T = T
        self.J = J
        self.lattice = np.random.choice([1, -1], size=L*L).astype(np.int8)

    def run_simulation(self, sweeps):
        if cython_metropolis_parallel is None:
            raise RuntimeError("Parallel Cython module is missing. Run setup.py first.")
        
        self.lattice = cython_metropolis_parallel(
            self.lattice, 
            float(self.T), 
            float(self.J), 
            int(sweeps),
            int(self.L)
        )

    def get_magnetization(self):
        return np.mean(self.lattice)
    

def run_and_store(self, total_steps, interval=1000):
    history = []
    for s in range(0, total_steps, interval):
        self.run_simulation(interval)
        history.append(self.lattice.copy())
    np.savez_compressed('data/output/sim_history.npz', *history)
    print(f"Saved {len(history)} frames to sim_history.npz")