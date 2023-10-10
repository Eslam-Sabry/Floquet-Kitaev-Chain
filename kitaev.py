from functools import reduce
import numpy as np
from constants import *
from hamiltonian import *
from tqdm import tqdm
from floquet import *
from scipy.integrate import quad


def onsite(_mu):
    return -_mu * s_z


def offsite(_t, _delta):
    return -_t * s_z + 1j * _delta * s_y