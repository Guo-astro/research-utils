import numpy as np
import matplotlib.pyplot as plt


def analytic_sod_strong_solution_1d(x, t, gamma=1.666666666, x0=0.0):
    """
    Analytic solution for the shock tube problem using simulation initial conditions.

    Left state:  rho_l = 1.0,   p_l = 1.0,    u_l = 0.0
    Right state: rho_r = 0.25,  p_r = 0.1795, u_r = 0.0
    x0: initial discontinuity location (default 0.5)

    Parameters:
        x : spatial positions (numpy array)
        t : time (t > 0)
        gamma : ratio of specific heats
        x0 : position of the initial discontinuity

    Returns:
        rho, p, u: density, pressure, and velocity arrays evaluated at x and t.
    """
    # Left state
    rho_l, p_l, u_l = 1.0, 10000.0, 0.0
    # Right state (from your simulation)
    rho_r, p_r, u_r = 1.0, 0.0001, 0.0

    # Sound speeds
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Handle t = 0 case
    if t == 0:
        rho = np.where(x < x0, rho_l, rho_r)
        p = np.where(x < x0, p_l, p_r)
        u = np.zeros_like(x)
        return rho, p, u

    # Functions for Newton-Raphson iteration to solve for p_star
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

    # Initial guess for p_star and iterative solution
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

    # Star region velocity (contact discontinuity speed)
    f_l = f(p_star, rho_l, p_l, c_l)
    f_r = f(p_star, rho_r, p_r, c_r)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)

    # Star region density
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

    # Self-similar variable (accounting for initial discontinuity x0)
    xi = (x - x0) / t

    # Speeds in the solution
    xi_head = u_l - c_l  # head of left rarefaction
    c_star_l = c_l * (p_star / p_l) ** ((gamma - 1) / (2 * gamma))
    xi_tail = u_star - c_star_l  # tail of left rarefaction

    if p_star > p_r:
        S_r = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_r) +
                                  (gamma - 1) / (2 * gamma))
    else:
        S_r = u_r + c_r

    # Allocate arrays for solution
    rho = np.zeros_like(x)
    p = np.zeros_like(x)
    u = np.zeros_like(x)

    # Region 1: Left undisturbed state
    mask1 = xi < xi_head
    rho[mask1] = rho_l
    p[mask1] = p_l
    u[mask1] = u_l

    # Region 2: Left rarefaction fan
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
    mask4 = (xi >= u_star) & (xi < S_r)
    rho[mask4] = rho_star_r
    p[mask4] = p_star
    u[mask4] = u_star

    # Region 5: Right undisturbed state
    mask5 = xi >= S_r
    rho[mask5] = rho_r
    p[mask5] = p_r
    u[mask5] = u_r

    return rho, p, u


# # Define spatial domain and time
# x = np.linspace(0, 1, 500)  # simulation domain from 0 to 1
# t = 0.5  # time at which solution is evaluated
# gamma = 1.4
# x0 = 0.5  # initial discontinuity is at x=0.5
#
# # Compute the analytic solution with the updated conditions
# rho, p, u = analytic_sod_solution_1d(x, t, gamma, x0)
#
# # Plotting the analytic solution
# fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
#
# axs[0].plot(x, rho, 'k-', lw=2)
# axs[0].set_ylabel('Density')
# axs[0].set_title(f'Analytic Shock Tube at t = {t}')
#
# axs[1].plot(x, p, 'k-', lw=2)
# axs[1].set_ylabel('Pressure')
#
# axs[2].plot(x, u, 'k-', lw=2)
# axs[2].set_ylabel('Velocity')
# axs[2].set_xlabel('x')
#
# plt.tight_layout()
# plt.show()
