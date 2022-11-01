from matplotlib import pyplot as plt

from constants import *
from scipy import linalg as sla
import matplotlib as mp
import numpy.linalg as la

# plt.style.use('seaborn')		# Setting the plotting style
mp.rcParams['figure.figsize'] = (20, 10)  # Setting the size of the plots


def one_period_propagator(hamiltonians, T, n=None):  # TODO: Return this to work in the general case
    if isinstance(hamiltonians, list):
        n = len(hamiltonians)
    elif callable(hamiltonians):
        hamiltonians = [hamiltonians(i * (T / n)) for i in range(n)]
    exps = [sla.expm(-1j * h * T / n) for h in hamiltonians]
    return np.matmul(exps[0], exps[1])


def floquet_hamiltonian(hamiltonians, T, n=None):
    u_T = one_period_propagator(hamiltonians, T, n)
    eigenval, eigenvec = la.eig(u_T)
    h_f1 = eigenvec @ np.diag(np.log(eigenval) * 1j / np.pi) @ np.conj(eigenvec.transpose())
    return h_f1
    # return 1j * sla.logm(u_T)/np.pi    #this is slow. better diagonalize then take the logarithm


def plot_spectrum(Hmat, N=25):
    evals, evecs = la.eigh(Hmat)
    evals = evals.real
    plt.scatter(np.arange(len(evals)), evals)
    plt.title('Energy Spectrum of Chain with {} Sites'.format(N))
    plt.show()
    # plt.savefig("spectrum")

