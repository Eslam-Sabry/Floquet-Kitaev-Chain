from kitaev import *
from tqdm import tqdm

N_global = 100
d_global = 2
h_1 = Hamiltonian(d=d_global, N=N_global, u=onsite(_mu=0 * t), v=offsite(_delta=1 * t, _t=t))
h_2 = Hamiltonian(d=d_global, N=N_global, u=onsite(_mu=1 * t), v=offsite(_delta=1 * t, _t=t))
# var_T = np.linspace(0.2 / t, 5 / t, 100)  # What's what?
var_T = np.arange(0.02, 0.8, 0.02)
G_0_odd = []
G_0_even = []
G_pi_odd = []
G_pi_even = []

delta_amount = 1E-6
delta_positive_0 = (delta_amount * 1j) * np.eye(N_global * d_global)
delta_negative_0 = (-delta_amount * 1j) * np.eye(N_global * d_global)
delta_positive_pi = (1 + delta_amount * 1j) * np.eye(N_global * d_global)
delta_negative_pi = (-1 - delta_amount * 1j) * np.eye(N_global * d_global)

for j in tqdm(range(len(var_T))):
    h_f = floquet_hamiltonian([h_1.lattice_hamiltonian(), h_2.lattice_hamiltonian()], T=var_T[j])
    G_0_pos = np.array(la.inv(delta_positive_0 - h_f))
    G_0_neg = np.array(la.inv(delta_negative_0 - h_f))

    G_0_odd.append((G_0_pos - G_0_neg) / 2)
    G_0_even.append((G_0_pos + G_0_neg) / 2)

    G_pi_pos = np.array(la.inv(delta_positive_pi - h_f))
    G_pi_neg = np.array(la.inv(delta_negative_pi - h_f))
    G_pi_odd.append((G_pi_pos - G_pi_neg) / 2)
    G_pi_even.append((G_pi_pos + G_pi_neg) / 2)
    pass

f_0_odd = [(G_0_odd[i][0, 1].imag) for i in range(len(G_0_odd))]
f_0_even = [abs(G_0_even[i][0, 1].imag) for i in range(len(G_0_even))]
f_pi_odd = [abs(G_pi_odd[i][0, 1].imag) for i in range(len(G_pi_odd))]
f_pi_even = [abs(G_pi_even[i][0, 1].imag) for i in range(len(G_pi_even))]
print(len(G_0_odd))

plt.title("edge odd-w pairing at 0 energy of finite Kitaev chain ")
plt.plot(var_T, f_0_odd)
plt.ylabel('odd-w amplitude')
plt.xlabel('$period$')
# plt.yscale('log')
plt.legend()
plt.show()

plt.title("edge odd-w pairing at pi energy of finite Kitaev chain ")
plt.plot(var_T,f_pi_odd)
plt.ylabel('odd-w amplitude')
plt.xlabel('$period$')
#plt.yscale('log')
plt.legend()
plt.show()

plt.title("edge even-w pairing at zero energy of finite Kitaev chain ")
plt.plot(var_T, f_0_even)
plt.ylabel('even-w amplitude')
plt.xlabel('$period$')
# plt.yscale('log')
plt.legend()
plt.show()

plt.title("edge even-w pairing at pi energy of finite Kitaev chain ")
plt.plot(var_T,f_pi_even)
plt.ylabel('even-w amplitude')
plt.xlabel('$period$')
#plt.yscale('log')
plt.legend()
plt.show()