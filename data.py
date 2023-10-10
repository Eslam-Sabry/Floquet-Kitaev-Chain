from kitaev import *
from tqdm import tqdm
import pickle


print('starting')

t=1
N_global = 150
d_global = 2
mu1 = 0 * t 
mu2 = 1 * t 
delta1 = 1 * t 
delta2 = 1 * t 
t1 = 1 * t 
t2 = 1 * t
h_1 = Hamiltonian(d=d_global, N=N_global, u=onsite(_mu=mu1), v=offsite(_delta=delta1, _t=t1))
h_2 = Hamiltonian(d=d_global, N=N_global, u=onsite(_mu=mu2), v=offsite(_delta=delta2, _t=t2))
var_T = np.linspace(0.2, 5, 100)  
var_w = np.linspace(0,1,200)
Gs_odd = []
Gs_even = []

delta_amount =  1E-6
delta_positive_0 = (delta_amount * 1j) 
delta_negative_0 = (-delta_amount * 1j)
delta_positive_pi = (1 + delta_amount * 1j)
delta_negative_pi = (1 - delta_amount * 1j)

delta_positive_pi_2 = 0  +(-1 + delta_amount * 1j)
delta_negative_pi_2 = 0  +(-1 - delta_amount * 1j)

energiesv2 = []
zero_modes_nums = []
pi_modes_nums = []

for j in tqdm(range(len(var_T))):
    eval, evec = floquet_hamiltonian([h_1.lattice_hamiltonian(), h_2.lattice_hamiltonian()], T=var_T[j])

    eval = eval.real
    energiesv2.append(np.sort([(-1) ** n * val for n, val in enumerate(np.sort(np.abs(eval)))]))
    evec_inv = la.inv(evec)
    G_odd = []
    G_even = []
    
    for w in var_w:
        
        delta_positive = (w + delta_amount * 1j) 
        delta_negative = (w - delta_amount * 1j)

        if np.abs(w) < 1 - e_threshold:    
            G_pos = np.diag(1/(delta_positive - eval))
            G_neg = np.diag(1/(delta_negative - eval))
            G_odd.append(evec @ (G_pos - G_neg) @ evec_inv / 2)
            #G_even.append(evec @ (G_pos + G_neg) @ evec_inv / 2)

        else:
            delta_positive_2 = 0  +(-w + delta_amount * 1j)
            delta_negative_2 = 0  +(-w - delta_amount * 1j)
            G_pi_pos = np.diag(1/(delta_positive - eval) + 1/(delta_positive_2 - eval))
            G_pi_neg = np.diag(1/(delta_negative - eval) + 1/(delta_negative_2 - eval))
            G_odd.append(evec @ (G_pi_pos - G_pi_neg) @ evec_inv / 4)
            #G_even.append(evec @ (G_pi_pos + G_pi_neg) @ evec_inv / 4)
    
    #Gs_even.append(G_even)
    Gs_odd.append(G_odd)
    modes_ind,cnt_modes = check_modes(eval, 0)
    zero_modes_nums.append(cnt_modes)
    modes_ind,cnt_modes = check_modes(eval, 1)
    pi_modes_nums.append(cnt_modes)
    

    if (j+1)%20 == 0:
        np.save(f'Gs_odd{(j+1)/20}.npy', Gs_odd)
        Gs_odd = []
    pass



np.save('Gs_odd.npy', Gs_odd)
print('done')
"""
with open('Gs_odd','wb') as f: pickle.dump(Gs_odd, f)
with open('Gs_even','wb') as f: pickle.dump(Gs_even, f)"""