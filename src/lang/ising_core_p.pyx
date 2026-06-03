import numpy as np
cimport numpy as cnp
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel cimport prange
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cython_metropolis(cnp.int8_t[:] lattice, float T, float J, int sweeps, int L):
    cdef int x, y, i, idx, neighbors
    cdef float dE

    for i in range(sweeps):
        for x in prange(L, nogil=True):
            for y in range(x%2, L, 2):
                idx = x * L + y
                neighbors = (lattice[((x + 1) % L) * L + y] +
                    lattice[((x - 1 + L) % L) * L + y] + 
                    lattice[x * L + ((y + 1) % L)] + 
                    lattice[x * L + ((y - 1 + L) % L)])
                dE = 2.0 * J * lattice[idx] * neighbors
                if dE < 0 or (rand() / <float>RAND_MAX) < exp(-dE / T):
                    lattice[idx] *= -1
        for x in prange(L, nogil=True):
            for y in range(1-(x%2), L, 2):
                idx = x * L + y
                neighbors = (lattice[((x + 1) % L) * L + y] +
                    lattice[((x - 1 + L) % L) * L + y] + 
                    lattice[x * L + ((y + 1) % L)] + 
                    lattice[x * L + ((y - 1 + L) % L)])
                dE = 2.0 * J * lattice[idx] * neighbors
                if dE < 0 or (rand() / <float>RAND_MAX) < exp(-dE / T):
                    lattice[idx] *= -1
    return np.asarray(lattice)