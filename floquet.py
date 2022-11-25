from functools import reduce
import imp
import numpy as np
from constants import *
from green_function import *
#import holoviews
from scipy import linalg as sla
import matplotlib as mp
#plt.style.use('seaborn')		# Setting the plotting style
mp.rcParams['figure.figsize'] = (20, 10)  # Setting the size of the plots
import numpy.linalg as la 
from scipy.integrate import quad


def one_period_propagator(hamiltonians, T, n=None):
    if isinstance(hamiltonians, list):
        num = len(hamiltonians)
        exps = [sla.expm(-1j * h * T / num) for h in hamiltonians]
        return reduce(np.matmul, exps)

    elif callable(hamiltonians):
        hamiltonian = [hamiltonians(i*(T/n)) for i in range(n)]
        exps = [sla.expm(-1j * h * T / n) for h in hamiltonian]
        return reduce(np.matmul, exps)

def floquet_hamiltonian(hamiltonians,T,n=None):
    u_T = one_period_propagator(hamiltonians,T,n)
    eval, evec= la.eig(u_T)
    h_f1 =evec @ np.diag(np.log(eval)*1j/np.pi) @ la.inv(evec)
    return h_f1
    #return 1j * sla.logm(u_T)/np.pi    #this is slow. better diagonalize then take the logarithm 


#فكرة فاشلة
#def propagator(hamiltonian,time,N,d,order=10,**kwargs):
#    u = lambda x: np.eye(d*N,dtype=complex)
#    for i in range(order):
#        u = lambda x: u(x) - i * np.vectorize(quad)(lambda t: hamiltonian(t,kwargs)@u(t), 0, x)
#    return u(time)



def calculate_finite_spectrum(periods, hamiltonians):
    energies = []
    for T in periods:
        U = one_period_propagator(hamiltonians, T)
        phases = np.angle(la.eigvals(U))
        phases = np.sort(np.abs(phases))
        ev = np.sort([(-1) ** n * val for n, val in enumerate(phases)])
        energies.append(ev)
    return np.array(energies).real


def calculate_bands(momenta, hamiltonians_k, T):
    energies = []
    for k in momenta:
        hamiltonians = [h_k(k) for h_k in hamiltonians_k]
        U = one_period_propagator(hamiltonians, T)
        phases = np.angle(la.eigvals(U))
        phases = np.sort(np.abs(phases))
        ev = np.sort([(-1) ** n * val for n, val in enumerate(phases)])
        energies.append(ev)
    return np.array(energies).real

def plot_spectrum(Hmat,N=N):
    evals,evecs = la.eigh(Hmat)
    evals = evals.real
    plt.scatter(np.arange(len(evals)),evals)
    plt.title('Energy Spectrum of Chain with {} Sites'.format(N))
    plt.show()
    #plt.savefig("spectrum")


e_threshold = 1E-6


def check_modes(evals, mode_energy,e_threshold=None):
    nmodes = 0
    if e_threshold == None:
        modes_ind = np.where(np.isclose(abs(evals), mode_energy * np.ones(len(evals))))[0]
    else:
        modes_ind = np.where(np.isclose(abs(evals), mode_energy * np.ones(len(evals)),atol=e_threshold))[0]
    return modes_ind,len(modes_ind)
    

def plot_modes(evals,evecs, mode_energy,e_threshold=None):
    modes_ind,cnt_modes = check_modes(evals, mode_energy,e_threshold=e_threshold)
    if cnt_modes > 0:
        fig,ax = plt.subplots(1,cnt_modes,figsize=(20, 10))
        fig.suptitle('Probability distribution of $E = \pm {}$ modes'.format(mode_energy),fontsize=20, fontweight='bold')
        for cnt in range(cnt_modes):
            ax1 = ax[cnt]
            ax1.plot(np.abs(evecs[:,modes_ind[cnt]])**2)
            ax1.set_title('Edge mode {}'.format(cnt+1),fontsize=20)
            ax1.set_xlabel('Site Number',fontsize=20)
            ax1.set_ylabel('$|\psi|^2$',fontsize=20)
            #ax1.text(0.43, 0.95, param_info, transform=ax1.transAxes, fontsize=16,
        #verticalalignment=('top', bbox=dict(boxstyle="square",facecolor="white"))
            ax1.tick_params(axis='both', which='major', labelsize=16)
        #plt.savefig('Edge_modes_Kitaev.pdf')
        plt.show()


