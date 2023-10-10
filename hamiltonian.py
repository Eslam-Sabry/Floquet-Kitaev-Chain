import torch
import torch.linalg as la
from matplotlib import pyplot as plt
import math
# from constants import *

class Hamiltonian():
    def __init__(self, d: int, N: int, u, v):
        # d = degrees of freedom
        self.d = d
        # number of lattice points
        self.N = N
        # same site interaction
        self.u = u
        # nearest neighbors interaction
        self.v = v

    def lattice_hamiltonian(self, *args, **kwargs):
        u = self.u(*args, **kwargs) if callable(self.u) else self.u
        v = self.v(*args, **kwargs) if callable(self.v) else self.v
        d, N = self.d, self.N
        H = torch.zeros([d * N, d * N], dtype=torch.complex64, device=device, requires_grad=False)
        for i in range(N - 1):
            H[i * d:(i + 1) * d, i * d:(i + 1) * d] = u
            H[i * d:(i + 1) * d, (i + 1) * d:(i + 2) * d] = v
            H[(i + 1) * d:(i + 2) * d, i * d:(i + 1) * d] = torch.conj(v.transpose(0,1))
        H[(N - 1) * d:N * d, (N - 1) * d:N * d] = u
        return H

    def k_space_hamiltonian(self, k):
        d, N, u, v = self.d, self.N, self.u, self.v
        v_dagger = torch.conj(v.transpose(0,1))
        v_sym = (v + v_dagger) / 2
        v_asym = (v - v_dagger) / (2)
        k = k[:, None, None]
        H_k = u + 2 * torch.cos(k) * v_sym - 1j * 2 * torch.sin(k) * v_asym
        return H_k

    def plot_spectrum(self, *args, **kwargs):
        var_k = torch.linspace(-math.pi, math.pi, 50, dtype=torch.complex64, device=device, requires_grad=False)
        spectrum = []
        for i in range(len(var_k)):
            H_k = self.k_space_hamiltonian(k=var_k[i], *args, **kwargs)
            eval, evec = la.eigh(H_k)
            spectrum.append(eval[0].item())  # Convert tensor to a Python scalar for plotting
        plt.title("Energy spectrum")
        plt.plot(var_k.cpu(), spectrum)  # Move data back to CPU for plotting
        # plt.plot(var_mu/t,G_img-F_img,label = 'G_img-F_img')
        plt.ylabel('E')
        plt.xlabel('$k$')
        plt.show()


# Assuming you have already initialized the device as mentioned earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage:
# Replace u and v with PyTorch tensors or functions that return PyTorch tensors
# u = torch.tensor(...)
# v = torch.tensor(...)
# hamiltonian = Hamiltonian(d=..., N=..., u=u, v=v)
# hamiltonian.plot_spectrum(...)
