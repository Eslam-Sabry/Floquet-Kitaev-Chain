from functools import reduce
import numpy as np
from constants import *
from green_function import *
from tqdm import tqdm
from scipy import linalg as sla
import matplotlib as mp
plt.style.use('seaborn')		# Setting the plotting style
mp.rcParams['figure.figsize'] = (15, 10)  # Setting the size of the plots
import numpy.linalg as la 
from floquet import *
from scipy.integrate import quad


def onsite(mu=mu, **kwargs):
    return -mu * s_z


def offsite(t=t, delta=delta,**kwargs):
    return -t * s_z +  1j * delta * s_y


N = 100

periods = np.linspace(0.2 / t, 5 / t, 100)
momenta = np.linspace(-2*np.pi, 2*np.pi,100)

Delta = 1*t

h_1 = Hamiltonian(d,N,onsite(mu=0*t),offsite(delta=Delta))
h_2 = Hamiltonian(d,N,onsite(mu=1*t),offsite(delta=Delta))

h = Hamiltonian(d,N,onsite,offsite)



#a = np.array([[np.sin,np.abs],[np.cos,lambda x: 1]])
#ans = np.vectorize(quad)(a,0,np.pi)
#print(ans[0])
T = 0.5
def kitaev_hamiltonian(time,period = T,h_1=h_1,h_2=h_2,**kwargs):
    if time < period/2:
        return h_1.lattice_hamiltonian()
    else:
        return h_2.lattice_hamiltonian()


#u = propagator(hamiltonian, T,N,d)

def kitaev_propagator(time,period = T, h_1=h_1,h_2=h_2):
    if time <= period/2:
        return sla.expm(-1j*h_1.lattice_hamiltonian()*time)
    elif period >= time > period/2:
        exps = [sla.expm(-1j * h_1.lattice_hamiltonian() * (time - period / 2)) ,sla.expm(-1j * h_2.lattice_hamiltonian() * period / 2)]
        #return reduce(np.matmul, exps)
        return np.matmul(sla.expm(-1j * h_1.lattice_hamiltonian() * (time - period / 2)) ,sla.expm(-1j * h_2.lattice_hamiltonian() * period / 2))

def kitaev_kick_operator(time,period = T, h_1 = h_1, h_2 = h_2):
    U = kitaev_propagator(time,period=period,h_1=h_1,h_2=h_2)
    h_f = floquet_hamiltonian([h_1.lattice_hamiltonian(),h_2.lattice_hamiltonian()],T=period)
    p = np.matmul(U,sla.expm(1j*np.pi*h_f*(time/period)))
    return p
    
#print(kitaev_propagator(T))

#h_f = floquet_hamiltonian([h_1.lattice_hamiltonian(),h_2.lattice_hamiltonian()],T=2*np.pi)
times = np.linspace(0,T, 100)
#eval, evec = la.eigh(h_f)
p = [kitaev_kick_operator(t) for t in tqdm(times)]
#pi_majorana = [np.dot(pp,eval[0]) for pp in p]
"""plot_spectrum(h_f)
print(evec[:,0].size)
plt.plot(np.arange(evec[:,0].size),np.abs(evec[:,0])**2)
plt.show()"""
print("finish")
