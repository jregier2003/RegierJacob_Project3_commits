import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

Mu_e = 2
RHO_0 = 9.74 * 10**5 * Mu_e
M_0 = 5.67 * 10**33 / Mu_e**2
R_0 = 7.72 * 10**8 / Mu_e


def system_of_equations(r, ystate):
    rho, m = ystate
    if rho <= 0:  
        return [0, 0]
    x = rho**(1/3)
    gamma_x = x**2 / (3 * (1 + x**2)**0.5)
    dystatedr = [-m * rho / (gamma_x * r**2), rho * r**2]
    return dystatedr

def density_equals_zero(r, ystate):
    Density_Tol = 1e-2
    return ystate[0] - Density_Tol

density_equals_zero.terminal = True
density_equals_zero.direction = -1