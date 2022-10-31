from functools import reduce
import numpy as np
from constants import *
from green_function import *
from tqdm import tqdm
from scipy import linalg as sla
import matplotlib as mp

import numpy.linalg as la
from floquet import *
from scipy.integrate import quad
from scipy import fft

plt.style.use('seaborn')  # Setting the plotting style
mp.rcParams['figure.figsize'] = (15, 10)  # Setting the size of the plots


# TODO: Name shadowing may cause issues.
def onsite(mu=mu):
    return -mu * s_z


# TODO: Name shadowing may cause issues.
def offsite(t=t, delta=delta):
    return -t * s_z + 1j * delta * s_y


N = 2

periods = np.linspace(0.2 / t, 5 / t, 100)
momenta = np.linspace(-2 * np.pi, 2 * np.pi, 100)

Delta = 1 * t

h_1 = Hamiltonian(d, N, onsite(mu=0 * t), offsite(delta=Delta))
h_2 = Hamiltonian(d, N, onsite(mu=1 * t), offsite(delta=Delta))

h = Hamiltonian(d, N, onsite, offsite)

# TODO: Commented
# a = np.array([[np.sin,np.abs],[np.cos,lambda x: 1]])
# ans = np.vectorize(quad)(a,0,np.pi)
# print(ans[0])
T = 0.5


# TODO: Name shadowing may cause issues.
# TODO: This function is not used
def kitaev_hamiltonian(time, period=T, h_1=h_1, h_2=h_2):
    if time < period / 2:
        return h_1.lattice_hamiltonian()
    else:
        return h_2.lattice_hamiltonian()


# TODO: Commented
# u = propagator(hamiltonian, T,N,d)

# TODO: Name shadowing may cause issues.
def kitaev_propagator(time, period=T, h_1=h_1, h_2=h_2):
    if time % period <= period / 2:
        return sla.expm(-1j * h_1.lattice_hamiltonian() * time)
    else:
        # TODO: Unused variable
        exps = [sla.expm(-1j * h_1.lattice_hamiltonian() * (time - period / 2)),
                sla.expm(-1j * h_2.lattice_hamiltonian() * period / 2)]
        # TODO: Commented
        # return reduce(np.matmul, exps)
        return np.matmul(sla.expm(-1j * h_1.lattice_hamiltonian() * (time - period / 2)),
                         sla.expm(-1j * h_2.lattice_hamiltonian() * period / 2))


# TODO: Name shadowing may cause issues.
def kitaev_kick_operator(time, period=T, h_1=h_1, h_2=h_2):
    U = kitaev_propagator(time, period=period, h_1=h_1, h_2=h_2)
    h_f = floquet_hamiltonian([h_1.lattice_hamiltonian(), h_2.lattice_hamiltonian()], T=period)
    p = np.matmul(U, sla.expm(1j * np.pi * h_f * (time / period)))
    return p


print("hi")
