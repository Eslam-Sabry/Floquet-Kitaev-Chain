import numpy as np
from numpy import linalg as la
from hamiltonian import *
from constants import *


def mobius_transformation_matrix(z, h: Hamiltonian):
    # TODO: Name shadowing may cause issues.
    d, N, u, v = h.d, h.N, h.u, h.v
    v_dagger = np.conj(v.transpose())
    v_inv = la.inv(v)
    Id = np.eye(d)
    A = np.zeros([d, d])
    B = v_inv
    C = - v_dagger
    D = np.matmul(z * Id - u, v_inv)
    x = np.block([[A, B], [C, D]])
    return x


def sort_matrix(x, sort_by_func=np.absolute):
    # TODO: Name shadowing may cause issues.
    A, Q = la.eig(x)
    sorted_indices = np.argsort(sort_by_func(A))
    # TODO: Not used.
    A = A[sorted_indices]
    Q = Q[:, sorted_indices]
    return Q


def left_green_function(z, h: Hamiltonian):
    # TODO: Name shadowing may cause issues.
    d = h.d
    x = mobius_transformation_matrix(z, h)
    Q = sort_matrix(x)
    B = Q[0:d, d:2 * d]
    D = Q[d:2 * d, d:2 * d]
    G_L = np.matmul(B, la.inv(D))
    return G_L
