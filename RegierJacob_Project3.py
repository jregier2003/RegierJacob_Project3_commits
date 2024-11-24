#commit 1
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

Mu_e = 2
RHO_0 = 9.74 * 10**5 * Mu_e
M_0 = 5.67 * 10**33 / Mu_e**2
R_0 = 7.72 * 10**8 / Mu_e