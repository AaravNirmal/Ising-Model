import numpy as np
import random
import math

def python_metropolis(lattice, T, J, total_steps):
    L = lattice.shape[0]
    for step in range(total_steps):
        i = random.randint(0, L-1)
        j = random.randint(0, L-1)
        s = lattice[i, j]
        neighbors_sum = (
            lattice[(i+1)%L, j] + 
            lattice[(i-1)%L, j] + 
            lattice[i, (j+1)%L] + 
            lattice[i, (j-1)%L]
        )
        delta_E = 2 * J * s * neighbors_sum
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            lattice[i, j] = -s
    return lattice