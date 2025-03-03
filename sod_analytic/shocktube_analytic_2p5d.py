# shocktube_analytic_2p5d.py

import numpy as np
import matplotlib.pyplot as plt


def analytic_sod_solution_2p5d(x, t, gamma=1.4, x0=0.5, dz=1.0, h_left=1.0, h_right=1.0):
    """
    Analytic solution for the Sod shock tube problem adjusted for a 2.5D DISPH simulation
    with different smoothing lengths on the left and right.

    Physical (1D) initial conditions:
      Left state:  ρₗ = 1.0,   pₗ = 1.0,    uₗ = 0.0
      Right state: ρᵣ = 0.25,  pᵣ = 0.1795, uᵣ = 0.0
    The discontinuity is located at x₀.

    In a 2.5D DISPH simulation each particle’s mass is computed as:
         m = ρ_phys · Δx · Δy · dz,
    and the density is computed with a normalized 3D cubic spline kernel. The integration
    of the kernel in the z–direction yields an extra factor of approximately 0.70/h.

    If different h are used on the left and right, then the simulation density is scaled by:
         F_left  = dz·(0.70/h_left)   (for the left side),
         F_right = dz·(0.70/h_right)  (for the right side).

    To compare the analytic solution with simulation output, we multiply the 1D analytic
    density and pressure by a piecewise scaling factor:
         use F_left when (x-x₀)/t < u* and F_right otherwise,
    where u* is the star-region velocity.

    Parameters:
      x       : 1D numpy array of spatial positions along x.
      t       : time (t > 0).
      gamma   : ratio of specific heats (default 1.4).
      x0      : initial discontinuity location.
      dz      : effective layer thickness used in the simulation.
      h_left  : smoothing length used in the left region.
      h_right : smoothing length used in the right region.

    Returns:
      (ρ, p, u) : numpy arrays of density, pressure, and velocity at positions x and time t,
                  with density and pressure scaled piecewise.
    """
    # Physical states:
    rho_l, p_l, u_l = 1.0, 1.0, 0.0
    rho_r, p_r, u_r = 0.25, 0.1795, 0.0

    # Sound speeds:
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    if t == 0:
        rho = np.where(x < x0, rho_l, rho_r)
        p = np.where(x < x0, p_l, p_r)
        u = np.zeros_like(x)
        F_left = dz * (0.70 / h_left)
        F_right = dz * (0.70 / h_right)
        scaling = np.where(x < x0, F_left, F_right)
        return rho * scaling, p * scaling, u

    # Newton-Raphson functions for p_star.
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

    # Solve for p_star:
    p_star = 0.5 * (p_l + p_r)
    tol = 1e-6
    max_iter = 100
    for i in range(max_iter):
        f_total = f(p_star, rho_l, p_l, c_l) + f(p_star, rho_r, p_r, c_r) + (u_r - u_l)
        df_total = df_dp(p_star, rho_l, p_l, c_l) + df_dp(p_star, rho_r, p_r, c_r)
        p_new = p_star - f_total / df_total
        if abs(p_new - p_star) < tol:
            p_star = p_new
            break
        p_star = p_new

    # Star region velocity:
    f_l = f(p_star, rho_l, p_l, c_l)
    f_r = f(p_star, rho_r, p_r, c_r)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)

    # Star region densities:
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

    # Self-similar variable:
    xi = (x - x0) / t

    # Allocate arrays:
    rho = np.zeros_like(x)
    p = np.zeros_like(x)
    u = np.zeros_like(x)

    # Region 1: Left undisturbed (xi < xi_head)
    xi_head = u_l - c_l
    mask1 = xi < xi_head
    rho[mask1] = rho_l
    p[mask1] = p_l
    u[mask1] = u_l

    # Region 2: Left rarefaction (xi_head <= xi < xi_tail)
    c_star_l = c_l * (p_star / p_l) ** ((gamma - 1) / (2 * gamma))
    xi_tail = u_star - c_star_l
    mask2 = (xi >= xi_head) & (xi < xi_tail)
    u[mask2] = 2 / (gamma + 1) * (c_l + xi[mask2])
    c_local = c_l - 0.5 * (gamma - 1) * u[mask2]
    rho[mask2] = rho_l * (c_local / c_l) ** (2 / (gamma - 1))
    p[mask2] = p_l * (c_local / c_l) ** (2 * gamma / (gamma - 1))

    # Region 3: Left star region (xi_tail <= xi < u_star)
    mask3 = (xi >= xi_tail) & (xi < u_star)
    rho[mask3] = rho_star_l
    p[mask3] = p_star
    u[mask3] = u_star

    # Region 4: Right star region (u_star <= xi < S_r)
    if p_star > p_r:
        S_r = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_r) +
                                  (gamma - 1) / (2 * gamma))
    else:
        S_r = u_r + c_r
    mask4 = (xi >= u_star) & (xi < S_r)
    rho[mask4] = rho_star_r
    p[mask4] = p_star
    u[mask4] = u_star

    # Region 5: Right undisturbed (xi >= S_r)
    mask5 = xi >= S_r
    rho[mask5] = rho_r
    p[mask5] = p_r
    u[mask5] = u_r

    # Compute piecewise scaling factors:
    F_left = dz * (0.70 / h_left)
    F_right = dz * (0.70 / h_right)
    # Use F_left if xi < u_star, otherwise F_right.
    scaling = np.where(xi < u_star, F_left, F_right)

    return rho * scaling, p * scaling, u


# Example usage for testing:
if __name__ == "__main__":
    x = np.linspace(0, 1, 500)  # spatial domain along x
    t = 0.5  # evaluation time
    dz = 1.0
    # For example, choose different h values for left and right:
    h_left = 0.038  # e.g., from left region mass/density relation
    h_right = 0.046  # e.g., from right region mass/density relation
    rho, p, u = analytic_sod_solution_2p5d(x, t, gamma=1.4, x0=0.0, dz=dz, h_left=h_left, h_right=h_right)

    # Plot the analytic solution.
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    axs[0].plot(x, rho, 'k-', lw=2)
    axs[0].set_ylabel('Density')
    axs[0].set_title(f'Analytic Shock Tube at t = {t} (2.5D adjusted, piecewise h)')

    axs[1].plot(x, p, 'k-', lw=2)
    axs[1].set_ylabel('Pressure')

    axs[2].plot(x, u, 'k-', lw=2)
    axs[2].set_ylabel('Velocity')
    axs[2].set_xlabel('x')

    plt.tight_layout()
    plt.show()
