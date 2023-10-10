import torch
import math

t = math.pi
delta = 0.50 * t
mu = 0.0 * t
B = 0.50 * t
a = 1.0 * t
omega = 1.0 * t
A = 0.01 * t
d = 2
N = 25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device, requires_grad=False)
s_x_np = s_x.cpu().numpy()
s_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device, requires_grad=False)
s_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device, requires_grad=False)
s_0 = torch.eye(2, dtype=torch.complex64, device=device, requires_grad=False)
j_x = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.complex64, device=device, requires_grad=False)
j_z = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=torch.complex64, device=device, requires_grad=False)
j_0 = torch.eye(3, dtype=torch.complex64, device=device, requires_grad=False)
