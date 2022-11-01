import numpy as np
import math


class Hamiltonian:
    def __init__(self, d: int, N: int, u, v):
        # d = degrees of freedom
        self.d = d
        # number of lattice points
        self.N = N
        # same site interaction
        self.u = u
        # nearest neighbors interaction
        self.v = v

    def lattice_hamiltonian(self):
        u = self.u() if callable(self.u) else self.u
        v = self.v() if callable(self.v) else self.v
        d, N = self.d, self.N
        H = np.zeros([d * N, d * N], complex)
        for i in range(N - 1):
            H[i * d:(i + 1) * d, i * d:(i + 1) * d] = u
            H[i * d:(i + 1) * d, (i + 1) * d:(i + 2) * d] = v
            H[(i + 1) * d:(i + 2) * d, i * d:(i + 1) * d] = np.conj(v.transpose())
        H[(N - 1) * d:N * d, (N - 1) * d:N * d] = u
        return H

    def k_space_hamiltonian(self, k):
        u = self.u() if callable(self.u) else self.u
        v = self.v() if callable(self.v) else self.v
        H_k = u + math.cos(k) * v.real + math.sin(k) * v.imag
        return H_k
