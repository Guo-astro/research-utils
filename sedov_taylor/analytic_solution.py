import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

##############################################################################
# 1) Solve the Sedov–Taylor ODE in dimensionless form (no top-hat approximation).
##############################################################################

def solve_sedov_selfsimilar(gamma=5/3, npts=200):
    """
    Numerically solve the self-similar ODE for the 3D Sedov–Taylor blast wave.
    Returns xi_arr, alpha_arr, phi_arr, beta_arr, where:
      - xi in [0, 1] is the dimensionless radius (xi=1 = shock).
      - alpha(xi) = rho / rho0
      - phi(xi)   = u / (Rdot)
      - beta(xi)  = p / [rho0 * (Rdot)^2]
    We integrate from xi=1 (shock) inward to xi=0.
    Reference: Landau & Lifshitz, Sedov's original work, etc.
    """

    alpha_shock = (gamma + 1)/(gamma - 1)
    phi_shock   = 2/(gamma + 1)
    beta_shock  = 2/(gamma + 1)

    # The constants from the jump conditions at xi=1
    C1 = alpha_shock * phi_shock
    C2 = alpha_shock*phi_shock**2 + beta_shock

    def calc_C3(alpha, phi, beta):
        # F(1) = phi * [0.5 alpha phi^2 + (gamma/(gamma-1))*beta]
        return phi*(0.5*alpha*(phi**2) + (gamma/(gamma-1))*beta)

    C3 = calc_C3(alpha_shock, phi_shock, beta_shock)

    def dphidxi(xi, phi):
        # near xi=0 or phi=0, avoid singularities
        if xi < 1e-12 or phi < 1e-12:
            return 0.0

        alpha_x = C1/(xi**2 * phi)
        beta_x  = (C2/(xi**2)) - alpha_x*(phi**2)

        # partial derivatives
        dalpha_dxi  = -2*C1/(xi**3*phi)
        dalpha_dphi = -C1/(xi**2 * (phi**2))

        dbeta_dxi   = -2*C2/(xi**3) - dalpha_dxi*(phi**2)
        dbeta_dphi  = -( dalpha_dphi*(phi**2) + 2*alpha_x*phi )

        # G = 0.5 alpha_x phi^2 + (gamma/(gamma-1))*beta_x
        G       = 0.5*alpha_x*(phi**2) + (gamma/(gamma-1))*beta_x
        dG_dxi  = 0.5*(dalpha_dxi*(phi**2)) + (gamma/(gamma-1))*dbeta_dxi
        dG_dphi = 0.5*(dalpha_dphi*(phi**2) + 2*alpha_x*phi) + (gamma/(gamma-1))*dbeta_dphi

        dF_dxi  = 2*xi*phi*G + (xi**2)*phi*dG_dxi
        dF_dphi = (xi**2)*(G + phi*dG_dphi)

        if abs(dF_dphi) < 1e-30:
            return 0.0
        return - dF_dxi / dF_dphi

    def odefun_reverse(xi, phi):
        # we integrate from xi=1 down to xi=1e-7, so reverse direction
        return - dphidxi(xi, phi[0])

    # Integrate from 1.0 down to 1e-7
    sol = solve_ivp(
        odefun_reverse,
        [1.0, 1e-7],
        [phi_shock],
        dense_output=True, max_step=1e-3, rtol=1e-7, atol=1e-9
    )

    # Evaluate at npts
    xi_grid = np.linspace(1.0, 1e-7, npts)
    phi_vals = sol.sol(xi_grid)[0]

    # Flip so xi goes from 0->1
    xi_arr = xi_grid[::-1]
    phi_arr = phi_vals[::-1]

    alpha_arr = np.zeros_like(phi_arr)
    beta_arr  = np.zeros_like(phi_arr)

    for i in range(npts):
        x  = xi_arr[i]
        ph = phi_arr[i] if phi_arr[i]>1e-30 else 1e-30
        if x<1e-12:
            alpha_arr[i] = alpha_shock
            beta_arr[i]  = beta_shock
        else:
            alpha_arr[i] = C1/(x**2 * ph)
            beta_arr[i]  = (C2/(x**2)) - alpha_arr[i]*(ph**2)

    return xi_arr, alpha_arr, phi_arr, beta_arr


##############################################################################
# 2) Class to evaluate the exact 3D Sedov–Taylor solution at (r, t).
##############################################################################

class SedovTaylorExact:
    """
    Precompute the dimensionless self-similar solution for gamma=5/3,
    then provide a method to evaluate (rho, p, vel, e) at arbitrary (r, t).
    """
    def __init__(self, gamma=5/3, npts=200):
        self.gamma = gamma
        self.npts  = npts
        # Solve once
        xi, alpha, phi, beta = solve_sedov_selfsimilar(gamma, npts)
        # Store
        self.xi_arr    = xi
        self.alpha_arr = alpha
        self.phi_arr   = phi
        self.beta_arr  = beta

    def __call__(self, r, t, E=1.0, rho0=1.0):
        """
        Evaluate the EXACT 3D Sedov–Taylor solution for each radius r at time t.
        Returns (rho, p, v, e) arrays of the same shape as r.
        """
        gamma = self.gamma
        if t < 1e-14:
            # Everything is basically ambient
            return (rho0*np.ones_like(r),
                    1e-10*np.ones_like(r),
                    np.zeros_like(r),
                    np.zeros_like(r))

        # Known constant for gamma=5/3
        kappa = 1.15167
        R_t = kappa * ((E * t**2)/rho0)**(1/5)
        # Shock speed ~ dR/dt
        Rdot_t = (2.0/5.0)*(R_t/t)

        xi_vals = r/(R_t + 1e-30)

        rho_out = np.empty_like(r)
        p_out   = np.empty_like(r)
        v_out   = np.empty_like(r)
        e_out   = np.empty_like(r)

        xi_sorted   = self.xi_arr
        alpha_sorted= self.alpha_arr
        phi_sorted  = self.phi_arr
        beta_sorted = self.beta_arr

        def alpha_of_xi(x):
            return np.interp(x, xi_sorted, alpha_sorted,
                             left=alpha_sorted[0], right=alpha_sorted[-1])
        def phi_of_xi(x):
            return np.interp(x, xi_sorted, phi_sorted,
                             left=phi_sorted[0], right=phi_sorted[-1])
        def beta_of_xi(x):
            return np.interp(x, xi_sorted, beta_sorted,
                             left=beta_sorted[0], right=beta_sorted[-1])

        for i in range(len(r)):
            xi_i = xi_vals[i]
            if xi_i >= 1.0:
                # outside shock
                rho_out[i] = rho0
                p_out[i]   = 1e-10
                v_out[i]   = 0.0
                e_out[i]   = p_out[i]/((gamma-1)*rho_out[i] + 1e-20)
            else:
                a = alpha_of_xi(xi_i)
                ph= phi_of_xi(xi_i)
                b = beta_of_xi(xi_i)
                rho_out[i] = rho0*a
                p_out[i]   = rho0*(Rdot_t**2)*b
                v_out[i]   = Rdot_t*ph
                e_out[i]   = p_out[i]/((gamma-1)*rho_out[i] + 1e-20)

        return rho_out, p_out, v_out, e_out


##############################################################################
# 3) Animate the Sedov–Taylor blast wave (SPH data + exact solution).
##############################################################################

def animate_sedov_exact(list_of_dataframes, list_of_times,
                        E=1.0, rho0=1.0, gamma=5 / 3, center=(0, 0, 0)):
    """
    Creates an animation of the SPH data (scatter) vs radius,
    overlayed with the exact 3D Sedov–Taylor solution (scatter),
    for each time frame in list_of_dataframes[0], list_of_times[0].
    """

    # Create figure with 4 subplots: P, rho, v, e
    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    var_names = ["Pressure [Pa]", "Density [kg/m^3]", "Velocity [m/s]", "Internal Energy [J/kg]"]
    for ax, var in zip(axes, var_names):
        ax.set_ylabel(var)
    axes[-1].set_xlabel("Radius [m]")

    # Create scatter objects for simulation
    scat_sim = {
        "pres": axes[0].scatter([], [], c="b", s=10, alpha=0.6, label="Sim"),
        "dens": axes[1].scatter([], [], c="b", s=10, alpha=0.6, label="Sim"),
        "vel": axes[2].scatter([], [], c="b", s=10, alpha=0.6, label="Sim"),
        "ene": axes[3].scatter([], [], c="b", s=10, alpha=0.6, label="Sim")
    }
    for ax in axes:
        ax.legend()

    # We'll create an instance of the solver
    # solver = SedovTaylorExact(gamma=gamma, npts=300)

    # We'll assume one set of dataframes/times => list_of_dataframes[0], list_of_times[0]
    df_list = list_of_dataframes[0]
    t_list = list_of_times[0]
    nframes = len(t_list)

    # ---------------------------------------------------------
    # Compute global y-limits for each variable across all frames.
    # ---------------------------------------------------------
    global_limits = {"pres": [np.inf, -np.inf],
                     "dens": [np.inf, -np.inf],
                     "vel": [np.inf, -np.inf],
                     "ene": [np.inf, -np.inf]}

    # Loop over all frames to update global min/max
    for df in df_list:
        # Positions and radii (for velocity projection)
        pos = df[["pos_x", "pos_y", "pos_z"]].values
        c = np.array(center, dtype=np.float64)
        dr = pos - c
        r = np.sqrt(np.sum(dr ** 2, axis=1))

        # Simulation data
        pres = df["pres"].values
        dens = df["dens"].values
        vel = df[["vel_x", "vel_y", "vel_z"]].values
        # Compute radial velocity
        v_r = np.sum(vel * dr, axis=1) / (r + 1e-20)
        # Compute internal energy
        ene = pres / (dens * (gamma - 1) + 1e-20)

        # Update limits (min and max)
        for key, data in zip(["pres", "dens", "vel", "ene"],
                             [pres, dens, v_r, ene]):
            global_limits[key][0] = min(global_limits[key][0], data.min())
            global_limits[key][1] = max(global_limits[key][1], data.max())

    # Optionally, add some padding to the limits
    pad = 0.1
    for key in global_limits:
        ymin, ymax = global_limits[key]
        yrange = ymax - ymin
        global_limits[key] = (ymin - pad * yrange, ymax + pad * yrange)

    # Set the computed y-limits to the corresponding axes
    axes[0].set_ylim([0,10])
    axes[1].set_ylim(global_limits["dens"])
    axes[2].set_ylim([0,4])
    axes[3].set_ylim([0,10])

    def init():
        # Nothing special to initialize; scatter is already empty
        return list(scat_sim.values())

    def update(frame_index):
        if frame_index >= nframes:
            return []

        df = df_list[frame_index]
        t = t_list[frame_index]

        # Extract positions, compute radius
        pos = df[["pos_x", "pos_y", "pos_z"]].values
        c = np.array(center, dtype=np.float64)
        dr = pos - c
        r = np.sqrt(np.sum(dr ** 2, axis=1))

        # Extract sim data
        dens = df["dens"].values
        pres = df["pres"].values
        vel = df[["vel_x", "vel_y", "vel_z"]].values
        v_r = np.sum(vel * dr, axis=1) / (r + 1e-20)
        ene = pres / (dens * (gamma - 1) + 1e-20)

        # Evaluate exact solution at each r (if desired)
        # rho_ex, p_ex, v_ex, e_ex = solver(r, t, E=E, rho0=rho0)

        # Update sim scatter offsets
        scat_sim["pres"].set_offsets(np.column_stack((r, pres)))
        scat_sim["dens"].set_offsets(np.column_stack((r, dens)))
        scat_sim["vel"].set_offsets(np.column_stack((r, v_r)))
        scat_sim["ene"].set_offsets(np.column_stack((r, ene)))

        # Adjust x-limits based on the current frame data
        rmax = r.max() if len(r) > 0 else 1.0
        for ax in axes:
            ax.set_xlim(0, rmax)
            # (We already set the y-limits globally.)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)  # only update x-axis

        return list(scat_sim.values())

    ani = FuncAnimation(fig, update, frames=nframes,
                        init_func=init, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani


##############################################################################
# 4) Example main that loads multiple frames, then animates them.
##############################################################################

def main():
    from main import load_dataframes_3d

    data_dirs = [
        "/Users/guo/OSS/sphcode/sample/sedov_taylor/results/GSPH/sedov_taylor/3D",
    ]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_3d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)

    # Explicitly check the first frame:
    first_df = list_of_dataframes[0][0]
    pos = first_df[["pos_x", "pos_y", "pos_z"]].values
    print("First frame particle positions range:", pos.min(), pos.max())
    print("First frame density range:", first_df["dens"].min(), first_df["dens"].max())
    print("First frame pressure range:", first_df["pres"].min(), first_df["pres"].max())

    animate_sedov_exact(list_of_dataframes, list_of_times,
                        E=1.0, rho0=1.0, gamma=5/3, center=(0,0,0))

if __name__ == "__main__":
    main()
