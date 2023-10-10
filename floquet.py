from functools import reduce
import torch
import torch.linalg as la
import numpy as np
import matplotlib.pyplot as plt
from constants import *

def one_period_propagator(hamiltonians, T, n=None):
    if isinstance(hamiltonians, list):
        n = len(hamiltonians)
    elif callable(hamiltonians):
        hamiltonians = [hamiltonians(i * (T / n)) for i in range(n)]
    exps = []
    for h in hamiltonians:
        eval, evec = la.eigh(h)
        exps.append(torch.matmul(evec, torch.matmul(torch.diag_embed(torch.exp(eval * (-1j * T / n))), torch.conj(evec).transpose(-2,-1))))
    return reduce(torch.matmul, exps)



def floquet_hamiltonian(hamiltonians, T, n=None):
    u_T = one_period_propagator(hamiltonians, T, n)
    eigenval, eigenvec = la.eig(u_T)
    return (torch.log(eigenval) * 1j / np.pi) , eigenvec


def floquet_hamiltonian2(hamiltonians, T, n=None):
    u_T = one_period_propagator(hamiltonians, T, n)
    eigenval, eigenvec = la.eig(u_T)
    h_f1 = torch.matmul(eigenvec, torch.matmul(torch.diag_embed(torch.log(eigenval) * 1j / np.pi), la.inv(eigenvec)))
    return h_f1


def calculate_finite_spectrum(periods, hamiltonians):
    energies = []
    for T in periods:
        U = one_period_propagator(hamiltonians, T)
        phases = torch.angle(la.eigvals(U))
        phases = torch.sort(torch.abs(phases))
        ev = torch.sort(torch.stack([(-1) ** n * val for n, val in enumerate(phases)]))
        energies.append(ev)
    return torch.stack(energies).real



def calculate_bands(momenta, hamiltonians_k, T):
    energies = []
    hamiltonians = [h_k(momenta) for h_k in hamiltonians_k]
    U = one_period_propagator(hamiltonians, T)
    phases = torch.angle(la.eigvals(U))
    phases, indices = torch.sort(torch.abs(phases))
    ev = np.sort([(-1) ** n * val for n, val in enumerate(np.sort(np.abs(phases.cpu())))])
    #energies.append(ev)
    #return torch.stack(energies).real
    return ev.real



def plot_spectrum(Hmat, N=N):
    evals, evecs = la.eigh(Hmat)
    evals = evals.real
    plt.scatter(torch.arange(len(evals)).cpu(), evals.cpu())
    plt.title('Energy Spectrum of Chain with {} Sites'.format(N))
    plt.show()


e_threshold = 1E-6


def check_modes(evals, mode_energy, _e_threshold=None):
    nmodes = 0
    energy_matrix = mode_energy * torch.ones(len(evals),device=device)
    if _e_threshold is None:
        modes_ind = torch.where(torch.isclose(torch.abs(evals), energy_matrix))[0]
    else:
        modes_ind = torch.where(torch.isclose(torch.abs(evals), energy_matrix, atol=_e_threshold))[0]
    return modes_ind, len(modes_ind)
    

def plot_modes(evals, evecs, mode_energy, _e_threshold=None, save_figure=False, fig_name='title.pdf'):
    modes_ind, cnt_modes = check_modes(evals, mode_energy, _e_threshold=_e_threshold)
    if cnt_modes > 0:
        fig, ax = plt.subplots(1, cnt_modes, figsize=(20, 10))
        fig.suptitle('Probability distribution of $E = \pm {}$ modes'.format(mode_energy), fontsize=40, fontweight='bold')
        for cnt in range(cnt_modes):
            print(cnt)
            ax1 = ax[cnt] if cnt_modes != 1 else ax
            density = (torch.abs(evecs[:, modes_ind[cnt]]) ** 2).cpu()
            densityv2 = [density[2 * j] + density[2 * j + 1] for j in range(int(len(evals) / 2))]
            ax1.plot(densityv2)
            ax1.set_title('Edge mode {}'.format(cnt + 1))
            ax1.set_xlabel('Site Number')
            ax1.set_ylabel('$|\psi|^2$')
        if save_figure:
            plt.savefig(fig_name, bbox_inches='tight')
        plt.show()
        return modes_ind

