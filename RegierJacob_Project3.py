import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import pandas as pd

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


def solve_equations(rho_center):
    boundary_condition = [rho_center, 0]
    range_of_radius = [0.1, 1e4]
    solution = solve_ivp(
        system_of_equations,
        range_of_radius,
        boundary_condition,
        events=density_equals_zero
    )
    return solution


rho_center = 2.5e6
solution = solve_equations(rho_center)

radius = solution.t * R_0
mass = solution.y[1] * M_0

plt.plot(radius, mass, label=f'rho_c = {rho_center}')
plt.title('Mass vs Radius for White Dwarfs')
plt.xlabel('Radius (cm)')
plt.ylabel('Mass (g)')
plt.legend()
plt.show()


initial_densities = np.logspace(-1, 6.4, 10)
for rho_center in initial_densities:
    solution = solve_equations(rho_center)
    radius = solution.t * R_0
    mass = solution.y[1] * M_0

    #Convert to physical units
    radius_physical = solution.t * R_0
    mass_physical = solution.y[1] * M_0
    density_physical = solution.y[0] * RHO_0
    plt.plot(radius_physical, mass_physical, label=f'rho_c = {rho_center:.1e}')


plt.title('Mass vs Radius for Various Central Densities')
plt.xlabel('Radius (cm)')
plt.ylabel('Mass (g)')
plt.xscale('log')
plt.yscale('log')


#Chandrasekhar limit
M_Ch = 5.836 / Mu_e**2 
plt.axhline(y=M_Ch * 1.989e33, color='black', linestyle='--', label=f"Chandrasekhar Limit ({M_Ch:.2f} M_sun)")
plt.legend()
plt.show()



data = pd.read_csv('wd_mass_radius.csv')

obs_mass = data['M_Msun']  
obs_radius = data['R_Rsun']  
obs_mass_err = data['M_unc']  
obs_radius_err = data['R_unc']  

for rho_center in initial_densities:
    solution = solve_equations(rho_center)
    radius_physical = solution.t * R_0 / 6.957e10  
    mass_physical = solution.y[1] * M_0 / 1.989e33  
    plt.plot(radius_physical, mass_physical, label=f'rho_c = {rho_center:.1e}')

plt.errorbar(obs_radius, obs_mass, xerr=obs_radius_err, yerr=obs_mass_err, fmt='o', label='Observed Data')
plt.legend()
plt.yscale('log')
plt.xlabel('Radius (Solar Radii)')
plt.ylabel('Mass (Solar Masses)')
plt.title('Observed vs Calculated Mass-Radius Relationship')
plt.show()



rho_centers = [1e2, 1e5, 2.5e6]  
methods = ['RK45', 'LSODA', 'Radau']

for rho_center in rho_centers:
    plt.figure()
    boundary_condition = [rho_center, 0]
    for method in methods:
        solution = solve_ivp(
            system_of_equations,
            [0.1, 1e4],
            boundary_condition,
            method=method,
            events=density_equals_zero,
        )
        radius_physical = solution.t * R_0
        mass_physical = solution.y[1] * M_0
        plt.plot(radius_physical, mass_physical, label=f'Method: {method}')
    
    plt.legend()
    plt.title(f'Comparison of Methods for rho_c = {rho_center:.1e}')
    plt.xlabel('Radius (cm)')
    plt.ylabel('Mass (g)')
    plt.show()