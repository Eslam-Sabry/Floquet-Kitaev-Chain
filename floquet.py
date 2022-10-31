from functools import reduce
import numpy as np
from constants import *
from green_function import *
# import holoviews
from scipy import linalg as sla
import matplotlib as mp
import numpy.linalg as la
from scipy.integrate import quad

# plt.style.use('seaborn')		# Setting the plotting style
mp.rcParams['figure.figsize'] = (20, 10)  # Setting the size of the plots


def one_period_propagator(hamiltonians, T, n=None):
    if isinstance(hamiltonians, list):
        n = len(hamiltonians)
    elif callable(hamiltonians):
        hamiltonians = [hamiltonians(i * (T / n)) for i in range(n)]

    exps = [sla.expm(-1j * h * T / n) for h in hamiltonians]
    return reduce(np.matmul, exps)


def floquet_hamiltonian(hamiltonians, T, n=None):
    u_T = one_period_propagator(hamiltonians, T, n)
    eigenval, eigenvec = la.eig(u_T)
    h_f1 = eigenvec @ np.diag(np.log(eigenval) * 1j / np.pi) @ la.inv(eigenvec)
    return h_f1
    # return 1j * sla.logm(u_T)/np.pi    #this is slow. better diagonalize then take the logarithm


# فكرة فاشلة
# def propagator(hamiltonian,time,N,d,order=10,**kwargs):
#    u = lambda x: np.eye(d*N,dtype=complex)
#    for i in range(order):
#        u = lambda x: u(x) - i * np.vectorize(quad)(lambda t: hamiltonian(t,kwargs)@u(t), 0, x)
#    return u(time)


def calculate_finite_spectrum(periods, hamiltonians):
    energies = []
    # TODO: I need to understand what is going on here. This looks very strange.
    for T in periods:
        U = one_period_propagator(hamiltonians, T)
        phases = np.angle(la.eigvals(U))
        phases = np.sort(np.abs(phases))
        ev = np.sort([(-1) ** n * val for n, val in enumerate(phases)])
        energies.append(ev)
    return np.array(energies).real


def calculate_bands(momenta, hamiltonians_k, T):
    energies = []
    # TODO: I need to understand what is going on here. This looks very strange.
    for k in momenta:
        hamiltonians = [h_k(k) for h_k in hamiltonians_k]
        U = one_period_propagator(hamiltonians, T)
        phases = np.angle(la.eigvals(U))
        phases = np.sort(np.abs(phases))
        ev = np.sort([(-1) ** n * val for n, val in enumerate(phases)])
        energies.append(ev)
    return np.array(energies).real


# TODO: Name shadowing may cause issues.
def plot_spectrum(Hmat, N=N):
    evals, evecs = la.eigh(Hmat)
    evals = evals.real
    plt.scatter(np.arange(len(evals)), evals)
    plt.title('Energy Spectrum of Chain with {} Sites'.format(N))
    plt.show()
    # plt.savefig("spectrum")


# TODO: Not used!
e_threshold = 1E-6


# TODO: Name shadowing may cause issues. Does not default to outer e_threshold.
def check_modes(evals, mode_energy, e_threshold=None):
    # TODO: Unused variable
    nzmodes = 0
    if e_threshold is None:
        zmodes_ind = np.where(np.isclose(abs(evals), mode_energy * np.ones(len(evals))))[0]
    else:
        zmodes_ind = np.where(np.isclose(abs(evals), mode_energy * np.ones(len(evals)), atol=e_threshold))[0]
    return zmodes_ind, len(zmodes_ind)


# TODO: Name shadowing may cause issues. Does not default to outer e_threshold.
def plot_modes(evals, evecs, mode_energy, e_threshold=None):
    modes_ind, cnt_modes = check_modes(evals, mode_energy, e_threshold=e_threshold)
    if cnt_modes > 0:
        fig, ax = plt.subplots(1, cnt_modes, figsize=(20, 10))
        fig.suptitle('Probability distribution of $E = \\pm {}$ modes'.format(mode_energy), fontsize=20,
                     fontweight='bold')
        for cnt in range(cnt_modes):
            ax1 = ax[cnt]
            ax1.plot(np.abs(evecs[:, modes_ind[cnt]]) ** 2)
            ax1.set_title('Edge mode {}'.format(cnt + 1), fontsize=20)
            ax1.set_xlabel('Site Number', fontsize=20)
            ax1.set_ylabel('$|\\psi|^2$', fontsize=20)
            # ax1.text(0.43, 0.95, param_info, transform=ax1.transAxes, fontsize=16,
            # verticalalignment=('top', bbox=dict(boxstyle="square",facecolor="white"))
            ax1.tick_params(axis='both', which='major', labelsize=16)
        # plt.savefig('Edge_modes_Kitaev.pdf')
        plt.show()
