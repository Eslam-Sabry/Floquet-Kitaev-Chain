from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from kitaev import kitaev_kick_operator, T
from scipy import fft
from tqdm import tqdm


# TODO: Why is this here?
def f():
    pass


print(type(f))

print(-0.4 % 1)

# TODO: T is not defined. Do we want to import it from Kitaev? (T = 0.5)
# print(kitaev_propagator(T))

# h_f = floquet_hamiltonian([h_1.lattice_hamiltonian(),h_2.lattice_hamiltonian()],T=2*np.pi)
times = np.linspace(0, T, 100)
# eval, evec = la.eigh(h_f)
p = [kitaev_kick_operator(t) for t in tqdm(times)]
# pi_majorana = [np.dot(pp,eval[0]) for pp in p]
"""plot_spectrum(h_f)
print(evec[:,0].size)
plt.plot(np.arange(evec[:,0].size),np.abs(evec[:,0])**2)
plt.show()"""

p_fourier = fft.ifft(p)
# print(p)
# print(len(p_fourier))
# print(p_fourier)
# p_fourier_v2 = [np.vectorize(quad)(kitaev_propagator,-0.5*T,0.5*T) for n in range(-50,50)]

test = [np.array(
    [[np.sin(2 * np.pi * x / T), np.cos(2 * np.pi * x / T)], [-np.cos(2 * np.pi * x / T), np.sin(2 * np.pi * x / T)]])
        for x in range(-50, 50)]
testf = fft.ifft(test)
print(test)
print(testf)
test2 = np.cos(1 * 2 * np.pi * times / T)
testf2 = fft.ifft(test2)
print(testf2)
print("finish")
# freq = fft.fftfreq(times)
plt.plot(range(-50, 50), np.abs(testf2))
plt.grid()
plt.show()
