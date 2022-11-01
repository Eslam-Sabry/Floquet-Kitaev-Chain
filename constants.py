import numpy as np

t = np.pi  # seconds
mu = 0.0 * t
B = 0.50 * t
a = 1.0 * t
omega = 1.0 * t
A = 0.01 * t

s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
s_z = np.array([[1, 0], [0, -1]])
s_0 = np.eye(2)
j_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
j_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
j_0 = np.eye(3)
