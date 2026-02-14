import numpy as np
cimport numpy as cnp
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cython_metropolis(cnp.int8_t[:, :] lattice, float T, float J, int steps):
    cdef int L = lattice.shape[0]
    cdef int x, y, i, neighbors
    cdef float dE
    cdef int s_i

    for i in range(steps):
        x = rand() % L
        y = rand() % L
        s_i = lattice[x, y]

        neighbors = (lattice[(x + 1) % L, y] +
                     lattice[(x - 1 + L) % L, y] +
                     lattice[x, (y + 1) % L] +
                     lattice[x, (y - 1 + L) % L])
        
        dE = 2.0 * J * s_i * neighbors

        if dE < 0 or (rand() / float(RAND_MAX)) < exp(-dE / T):
            lattice[x, y] *= -1
            
    return np.asarray(lattice)