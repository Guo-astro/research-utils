#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

# --- Physical parameters ---
G = 1.0       # gravitational constant (code units)
gamma = 5/3   # adiabatic index
K_poly = 1.0  # polytropic constant
Sigma0 = 1.0  # central surface density, Σ(0)

# --- Radial grid ---
R_max = 10.0
num_points = 1000
R_array = np.linspace(1e-4, R_max, num_points)  # avoid R=0 singularity

def kernel(R, Rp):
    """
    Razor-thin disk kernel:
      K(R, Rp) = (2/π) * ellipk(m) / (R + Rp),
    where m = [2√(R*Rp)/(R+Rp)]².
    """
    if R + Rp < 1e-8:
        return 0.0
    m = (2.0 * np.sqrt(R * Rp) / (R + Rp))**2
    m = min(m, 0.9999)  # keep m in valid range
    return (2.0 / np.pi) * ellipk(m) / (R + Rp)

def compute_potential(R_array, Sigma_array, G):
    """
    Computes Φ(R) = -G ∫ Σ(R') R' dR' K(R,R'),
    using a direct loop + np.trapz (O(N^2) approach).
    """
    Phi = np.zeros_like(R_array)
    for i, R in enumerate(R_array):
        integrand = Sigma_array * R_array * np.array([kernel(R, Rp) for Rp in R_array])
        Phi[i] = -G * np.trapz(integrand, R_array)
    return Phi

# Initial guess for Σ(R): exponential, with Σ(0) = Sigma0
Sigma = Sigma0 * np.exp(-R_array / (R_max/5.0))
Sigma[0] = Sigma0

max_iter = 1000
tol = 1e-6
damping_factor = 0.1

# Lane–Emden iteration for γ=5/3:
# Σ(R)^(γ-1) = Σ(0)^(γ-1) - ((γ-1)/(γ*K_poly)) [Φ(R)-Φ(0)]
# => Σ(R) = ...
gamma_minus_1 = gamma - 1.0  # = 2/3
coeff = (gamma_minus_1)/(gamma * K_poly)  # = (2/3) / [(5/3)*K] = 2/(5*K)

for iteration in range(max_iter):
    Phi = compute_potential(R_array, Sigma, G)
    Phi0 = Phi[0]
    argument = Sigma0**(gamma_minus_1) - coeff * (Phi - Phi0)
    Sigma_new = np.power(np.maximum(argument, 0.0), 1.0 / gamma_minus_1)

    # Damping
    Sigma_new = damping_factor * Sigma_new + (1 - damping_factor) * Sigma
    diff = np.linalg.norm(Sigma_new - Sigma)
    print(f"Iteration {iteration+1}, diff = {diff:.4e}")
    if diff < tol:
        print("Converged!")
        Sigma = Sigma_new.copy()
        break
    Sigma = Sigma_new.copy()

# Export to CSV
def export_adiabatic_disk_csv(R_array, Sigma, filename="adiabatic_razor_thin_disk_sigma.csv"):
    data = np.column_stack((R_array, Sigma))
    try:
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["R", "Sigma"])
            writer.writerows(data)
        print(f"Exported to '{filename}' with {len(data)} rows.")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

export_adiabatic_disk_csv(R_array, Sigma)

# Plot for visual check
plt.figure()
plt.plot(R_array, Sigma, label='Converged Σ(R)')
plt.xlabel('R')
plt.ylabel('Surface Density Σ(R)')
plt.title('Razor-Thin Disk, γ=5/3, K=1.0, Σ0=1.0')
plt.legend()
plt.grid(True)
plt.show()
