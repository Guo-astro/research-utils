import numpy as np
import matplotlib.pyplot as plt

def analytic_sod_solution_2p5d(x, t, gamma=1.4, x0=0.0, dz=1.0, h_left=0.00325, h_right=0.007):
    """
    Analytic solution for the Sod shock tube problem adjusted for a 2.5D DISPH simulation
    matching the C++ initialization code.

    Physical initial conditions (from C++):
      Left state:  ρₗ = 1.0,  pₗ = 1.0,    uₗ = 0.0
      Right state: ρᵣ = 0.25, pᵣ = 0.1795, uᵣ = 0.0
    Discontinuity at x₀ = 0.0.

    SPH scaling:
      ρ_sim = ρ_phys * dz * (0.70 / h)
      h_left = 0.00325, h_right = 0.007 (from C++).

    Parameters:
      x       : 1D numpy array of spatial positions along x.
      t       : time (t > 0).
      gamma   : ratio of specific heats (default 1.4).
      x0      : initial discontinuity location.
      dz      : layer thickness (1.0 in C++).
      h_left  : smoothing length for left region (0.00325).
      h_right : smoothing length for right region (0.007).

    Returns:
      (ρ, p, u) : Scaled density, pressure, and velocity arrays.
    """
    # Calculate scaling factors
    F_left = dz * (0.70 / h_left)  # ~215.38
    F_right = dz * (0.70 / h_right)  # 100

    # Scaled initial conditions
    rho_l = 1.0 * F_left  # 215.38
    p_l = 1.0 * F_left  # 215.38
    u_l = 0.0
    rho_r = 0.25 * F_right  # 25
    p_r = 0.1795 * F_right  # 17.95
    u_r = 0.0

    # Run Riemann solver with scaled conditions
    # Placeholder: replace with actual solver

    # Sound speeds
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)


    # Newton-Raphson for p_star
    def f(p, rho, p_i, c):
        if p > p_i:
            A = 2.0 / ((gamma + 1) * rho)
            B = (gamma - 1) / (gamma + 1)
            return (p - p_i) * np.sqrt(A / (p + B * p_i))
        else:
            return (2 * c / (gamma - 1)) * ((p / p_i) ** ((gamma - 1) / (2 * gamma)) - 1)

    def df_dp(p, rho, p_i, c):
        if p > p_i:
            A = 2.0 / ((gamma + 1) * rho)
            B = (gamma - 1) / (gamma + 1)
            sqrt_term = np.sqrt(A / (p + B * p_i))
            return sqrt_term * (1 - 0.5 * (p - p_i) / (p + B * p_i))
        else:
            return (1 / (rho * c)) * (p / p_i) ** (-(gamma + 1) / (2 * gamma))

    # Solve for p_star
    p_star = 0.5 * (p_l + p_r)
    tol = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        f_total = f(p_star, rho_l, p_l, c_l) + f(p_star, rho_r, p_r, c_r) + (u_r - u_l)
        df_total = df_dp(p_star, rho_l, p_l, c_l) + df_dp(p_star, rho_r, p_r, c_r)
        p_new = p_star - f_total / df_total
        if abs(p_new - p_star) < tol:
            p_star = p_new
            break
        p_star = p_new

    # Star region velocity
    f_l = f(p_star, rho_l, p_l, c_l)
    f_r = f(p_star, rho_r, p_r, c_r)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)

    # Star region densities
    if p_star > p_l:
        B_l = (gamma - 1) / (gamma + 1)
        rho_star_l = rho_l * ((p_star / p_l + B_l) / (B_l * p_star / p_l + 1))
    else:
        rho_star_l = rho_l * (p_star / p_l) ** (1 / gamma)

    if p_star > p_r:
        B_r = (gamma - 1) / (gamma + 1)
        rho_star_r = rho_r * ((p_star / p_r + B_r) / (B_r * p_star / p_r + 1))
    else:
        rho_star_r = rho_r * (p_star / p_r) ** (1 / gamma)

    # Self-similar variable
    xi = (x - x0) / t

    # Allocate arrays
    rho = np.zeros_like(x)
    p = np.zeros_like(x)
    u = np.zeros_like(x)

    # Region 1: Left undisturbed
    xi_head = u_l - c_l
    mask1 = xi < xi_head
    rho[mask1] = rho_l
    p[mask1] = p_l
    u[mask1] = u_l

    # Region 2: Rarefaction
    c_star_l = c_l * (p_star / p_l) ** ((gamma - 1) / (2 * gamma))
    xi_tail = u_star - c_star_l
    mask2 = (xi >= xi_head) & (xi < xi_tail)
    u[mask2] = 2 / (gamma + 1) * (c_l + xi[mask2])
    c_local = c_l - 0.5 * (gamma - 1) * u[mask2]
    rho[mask2] = rho_l * (c_local / c_l) ** (2 / (gamma - 1))
    p[mask2] = p_l * (c_local / c_l) ** (2 * gamma / (gamma - 1))

    # Region 3: Left star region
    mask3 = (xi >= xi_tail) & (xi < u_star)
    rho[mask3] = rho_star_l
    p[mask3] = p_star
    u[mask3] = u_star

    # Region 4: Right star region
    if p_star > p_r:
        S_r = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_r) +
                                  (gamma - 1) / (2 * gamma))
    else:
        S_r = u_r + c_r
    mask4 = (xi >= u_star) & (xi < S_r)
    rho[mask4] = rho_star_r
    p[mask4] = p_star
    u[mask4] = u_star

    # Region 5: Right undisturbed
    mask5 = xi >= S_r
    rho[mask5] = rho_r
    p[mask5] = p_r
    u[mask5] = u_r

    # Apply SPH scaling
    F_left = dz * (0.70 / h_left)   # ≈ 215.38
    F_right = dz * (0.70 / h_right) # = 25.0
    scaling = np.where(xi < u_star, F_left, F_right)

    return rho * scaling, p * scaling, u

# Example usage
if __name__ == "__main__":
    x = np.linspace(-1, 1, 1000)  # Match C++ domain
    t = 0.2  # Typical time for Sod problem visualization
    rho, p, u = analytic_sod_solution_2p5d(x, t, gamma=1.4)

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    axs[0].plot(x, rho, 'k-', lw=2)
    axs[0].set_ylabel('Density')
    axs[0].set_title(f'Analytic Shock Tube at t = {t} (2.5D, h_left={0.00325}, h_right={0.007})')

    axs[1].plot(x, p, 'k-', lw=2)
    axs[1].set_ylabel('Pressure')

    axs[2].plot(x, u, 'k-', lw=2)
    axs[2].set_ylabel('Velocity')
    axs[2].set_xlabel('x')

    plt.tight_layout()
    plt.show()