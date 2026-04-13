High-Performance Ising Model Simulation:

    This repository contains a high-performance numerical simulation of the 2D Ising model, utilizing the Metropolis-Hastings algorithm to study phase transitions in ferromagnetic systems. The core computational engine is implemented in Cython to achieve near-C speeds while maintaining a Pythonic interface for data analysis.
    Technical Architecture: The project is structured to decouple the heavy numerical lifting from the user interface and visualization layers:
        1. Computational Core (src/ising_core.pyx): Written in Cython to bypass the Python Global Interpreter Lock (GIL). This module handles the intensive bit-level operations and energy calculations for the square lattice.
        2. Abstraction Layer (src/model.py): A high-level Python class that manages the simulation lifecycle, state transitions, and interfacing with the compiled C-binaries.
        3. Data Pipeline: Uses a structured CSV-based input system (data/input/) and generates automated lattice snapshots and magnetization plots (data/output/).
    Implementation Details: 
        The Metropolis Algorithm
        1. The simulation reaches thermal equilibrium by evaluating spin-flip probabilities based on the change in local energy $\Delta E$. The probability of a flip is given by: $$P = \min(1, e^{-\beta \Delta E})$$ Where $\beta = 1/k_B T$. At low temperatures, the system converges toward a state of long-range order, while high-temperature regimes exhibit entropy-driven disorder.
        Performance Optimization 
        1. To handle large lattice dimensions ($L > 100$) within a reasonable timeframe, the Metropolis loop was offloaded to Cython. By using static type declarations and C-level array access, the simulation achieves an execution speed roughly 80x faster than a pure NumPy implementation.
    Setup and Execution: 
        Requirements
        1. Python 3.8+
        2. C++ Build Tools (MSVC for Windows or GCC for Linux/macOS)
        3. Dependencies: numpy, matplotlib, pandas, cython 
    Compilation: The Cython source must be compiled into a machine-specific binary before the simulation can run which is done by writing the following into the terminal: python setup.py build_ext --inplace
    Usage: Configure simulation variables ($L, T, J, \text{steps}$) in data/input/parameters.csv.
        Run the simulation:Bashpython main.py
        The program outputs the final magnetization to the console and exports lattice visualizations to the output directory.
