Ising Model:

Project Structure
    The repository is organized to separate the computational logic from the analysis and visualization tools:

    main.py: The entry point for the application. It handles parameter loading, executes the simulation, and triggers result generation.

    setup.py: The build script used to compile the Cython core into a machine-compatible binary.

    src/: Contains the source code modules.

    ising_core.pyx: The Cython source file containing the performance-critical Metropolis algorithm.

    model.py: The high-level Python class that manages simulation states and interfaces with the compiled engine.

    visualization.py: Tools for generating lattice snapshots and plots.

    data/: Directory for simulation assets.

    input/: Contains parameters.csv to define simulation variables like temperature and grid size.

    output/: Stores generated images and simulation results.

Installation and Setup
    Environment: It is recommended to use a virtual environment. Install the necessary dependencies, including NumPy, Pandas, and Matplotlib.

    Compilation: Since the core engine is written in Cython, you must compile it on your local machine to generate the necessary .pyd or .so file. Run the following command from the root directory: python setup.py build_ext --inplace

    Compiler Requirements: On Windows, this requires the Microsoft C++ Build Tools with the Desktop development with C++ workload.

How to Run
    Define your simulation parameters in data/input/parameters.csv. Key parameters include:

    L: The linear dimension of the square lattice.

    T: The temperature of the system.

    J: The coupling constant.

    steps: The number of Monte Carlo steps to perform.

    Execute the simulation by running python main.py.

    The final magnetization will be printed to the terminal, and a visual representation of the final lattice state will be saved to the data/output/snapshots/ directory.

Theoretical Background
    The simulation uses the Metropolis-Hastings algorithm to reach thermal equilibrium. The probability of a spin flip depends on the change in energy, which is calculated based on the alignment of neighboring spins. At low temperatures, the system tends toward order (ferromagnetism), while at high temperatures, thermal fluctuations lead to a disordered state.

