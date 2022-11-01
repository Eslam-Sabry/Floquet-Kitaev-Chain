from floquet import *
from hamiltonian import Hamiltonian

plt.style.use('seaborn')  # Setting the plotting style
mp.rcParams['figure.figsize'] = (15, 10)  # Setting the size of the plots


def onsite(_mu):
    return -_mu * s_z


def offsite(_t, _delta):
    return -_t * s_z + 1j * _delta * s_y


h_1 = Hamiltonian(d=2, N=2, u=onsite(_mu=0 * t), v=offsite(_delta=1 * t, _t=t))
h_2 = Hamiltonian(d=2, N=2, u=onsite(_mu=1 * t), v=offsite(_delta=1 * t, _t=t))

T = 0.5

print("hi")
