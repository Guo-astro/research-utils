from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import math

from vis.common import load_dataframes_3d

# Global constant: first zero for n = 3/2.
XI1_GLOBAL = 3.65375


def compute_lane_emden_n32(num_steps=10000):
    """
    Compute the Lane–Emden solution for n = 3/2 using RK4 from xi = 1e-6 to XI1_GLOBAL.
    Returns:
      xi_arr: numpy array of xi values.
      theta_arr: numpy array of theta values.
    """
    xi = 1e-6
    theta = 1.0 - (xi ** 2) / 6.0
    dtheta = -xi / 3.0
    h = (XI1_GLOBAL - 1e-6) / num_steps

    xi_list = [xi]
    theta_list = [theta]

    for _ in range(num_steps):
        k1_theta = h * dtheta
        k1_dtheta = h * (- math.pow(theta, 1.5) - (2.0 / xi) * dtheta)

        k2_theta = h * (dtheta + k1_dtheta / 2)
        k2_dtheta = h * (- math.pow(theta + k1_theta / 2, 1.5) -
                         (2.0 / (xi + h / 2)) * (dtheta + k1_dtheta / 2))

        k3_theta = h * (dtheta + k2_dtheta / 2)
        k3_dtheta = h * (- math.pow(theta + k2_theta / 2, 1.5) -
                         (2.0 / (xi + h / 2)) * (dtheta + k2_dtheta / 2))

        k4_theta = h * (dtheta + k3_dtheta)
        k4_dtheta = h * (- math.pow(theta + k3_theta, 1.5) -
                         (2.0 / (xi + h)) * (dtheta + k3_dtheta))

        theta += (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6.0
        dtheta += (k1_dtheta + 2 * k2_dtheta + 2 * k3_dtheta + k4_dtheta) / 6.0
        xi += h

        xi_list.append(xi)
        theta_list.append(theta)

    return np.array(xi_list), np.array(theta_list)


def lane_emden_analytic_n32(r_max, dens_c, pres_c, num_steps=10000):
    """
    Compute the analytic Lane–Emden solution for n = 3/2.

    The solution is computed in xi-space (with xi from 1e-6 to XI1_GLOBAL) and then scaled:
      r = a * xi, with a = r_max / XI1_GLOBAL.

    The analytic density and pressure are given by:
      rho(r) = dens_c * theta^(3/2)
      P(r)   = pres_c  * theta^(5/2)

    Returns:
      r_analytic: radial positions.
      analytic_dens: density profile.
      analytic_pres: pressure profile.
    """
    xi_arr, theta_arr = compute_lane_emden_n32(num_steps=num_steps)
    # Scale xi to the physical radius (assume simulation r_max corresponds to xi = XI1_GLOBAL)
    a = r_max / XI1_GLOBAL
    r_analytic = xi_arr * a
    analytic_dens = dens_c * np.power(theta_arr, 1.5)
    analytic_pres = pres_c * np.power(theta_arr, 2.5)
    return r_analytic, analytic_dens, analytic_pres


def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Animate 3D scatter plots with separate 2D radial plots.
    Top row: 3D scatter (pos_x, pos_y, pos_z, colored by physics_key).
    Next four rows: Radial profiles for Density, Pressure, Velocity Magnitude, and Energy.

    Analytic Lane–Emden solutions (for n=3/2) are overlaid on the Density and Pressure plots.
    A time label is added at the top of the figure.
    """
    n_dirs = len(list_of_dataframes)
    # Five rows: one for 3D scatter and four for radial profiles.
    fig = plt.figure(figsize=(5 * n_dirs, 15))
    axes_3d = []

    # Create a global time label at the top of the figure.
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=16)

    # Lists for simulation and analytic lines.
    lines_dens = []
    lines_pres = []
    lines_vel = []
    lines_energy = []
    analytic_lines_dens = []
    analytic_lines_pres = []

    # --- 3D Scatter subplots ---
    for i, dataframes in enumerate(list_of_dataframes):
        ax3d = fig.add_subplot(5, n_dirs, i + 1, projection="3d")
        axes_3d.append(ax3d)
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
        pos_z_all = np.concatenate([df["pos_z"].values for df in dataframes])
        ax3d.set_xlim(pos_x_all.min() * 0.95, pos_x_all.max() * 1.05)
        ax3d.set_ylim(pos_y_all.min() * 0.95, pos_y_all.max() * 1.05)
        ax3d.set_zlim(pos_z_all.min() * 0.95, pos_z_all.max() * 1.05)
        ax3d.set_xlabel("pos_x [m]")
        ax3d.set_ylabel("pos_y [m]")
        ax3d.set_zlabel("pos_z [m]")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax3d.scatter([], [], [], s=1, c=[], cmap="viridis",
                            vmin=physics_all.min(), vmax=physics_all.max())
        title = (plot_titles[i] + " (3D)") if (plot_titles and i < len(plot_titles)) else f"Dir {i + 1}"
        ax3d.set_title(title)
        ax3d._scatter_obj = scat

    # --- Radial (2D) subplots ---
    radial_vars = [("dens", "Density", True),
                   ("pres", "Pressure", True),
                   ("vel", "Velocity Magnitude", False),
                   ("ene", "ene", False)]

    axes_intersect = {}
    for var, ylabel, analytic_avail in radial_vars:
        axes_intersect[var] = []
        # Map variable to subplot row.
        if var == "dens":
            row = 1
        elif var == "pres":
            row = 2
        elif var == "vel":
            row = 3
        elif var == "ene":
            row = 4
        for i in range(n_dirs):
            ax = fig.add_subplot(5, n_dirs, row * n_dirs + i + 1)
            ax.set_xlabel("Radius r [m]")
            ax.set_ylabel(ylabel)
            line_sim, = ax.plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
            line_analytic, = ax.plot([], [], linestyle='--', color='red', label='Analytic')
            axes_intersect[var].append(ax)
            if var == "dens":
                lines_dens.append(line_sim)
                analytic_lines_dens.append(line_analytic)
            elif var == "pres":
                lines_pres.append(line_sim)
                analytic_lines_pres.append(line_analytic)
            elif var == "vel":
                lines_vel.append(line_sim)
            elif var == "ene":
                lines_energy.append(line_sim)
            ax.legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        for ax in axes_3d:
            ax._scatter_obj._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line in (lines_dens + lines_pres + lines_vel + lines_energy +
                     analytic_lines_dens + analytic_lines_pres):
            line.set_data([], [])
        time_text.set_text('')
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel + lines_energy +
                analytic_lines_dens + analytic_lines_pres +
                [time_text])

    def update(frame_index):
        artists = []
        # Update the time label using the first dataset's time.
        current_time = list_of_times[0][frame_index]
        time_text.set_text(f"Time: {current_time:.3f} s")
        artists.append(time_text)

        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            # --- Update 3D scatter ---
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scat = axes_3d[i]._scatter_obj
            scat._offsets3d = (x, y, z)
            scat.set_array(df[physics_key].values)
            artists.append(scat)

            # --- Compute radial coordinate and sort ---
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            sort_idx = np.argsort(r)
            r_sorted = r[sort_idx]
            dens = df["dens"].values[sort_idx]
            pres = df["pres"].values[sort_idx]
            if "vel_z" in df.columns:
                vel = np.sqrt(df["vel_x"].values[sort_idx] ** 2 +
                              df["vel_y"].values[sort_idx] ** 2 +
                              df["vel_z"].values[sort_idx] ** 2)
            else:
                vel = np.sqrt(df["vel_x"].values[sort_idx] ** 2 +
                              df["vel_y"].values[sort_idx] ** 2)
            if "ene" in df.columns:
                energy = df["ene"].values[sort_idx]
            else:
                energy = np.full_like(r_sorted, np.nan)

            # --- Update simulation radial profiles ---
            lines_dens[i].set_data(r_sorted, dens)
            lines_pres[i].set_data(r_sorted, pres)
            lines_vel[i].set_data(r_sorted, vel)
            lines_energy[i].set_data(r_sorted, energy)

            axes_intersect["dens"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["dens"][i].set_ylim(dens.min(), dens.max())
            axes_intersect["pres"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["pres"][i].set_ylim(pres.min(), pres.max())
            axes_intersect["vel"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["vel"][i].set_ylim(vel.min(), vel.max())
            axes_intersect["ene"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["ene"][i].set_ylim(energy.min(), energy.max())

            # --- Compute analytic Lane–Emden (n=3/2) profile ---
            if r_sorted.size > 0:
                r_max_sim = r_sorted.max()
                dens_c = dens[0]  # approximate central density
                pres_c = pres[0]  # approximate central pressure
                r_analytic, analytic_dens, analytic_pres = lane_emden_analytic_n32(r_max_sim, dens_c, pres_c)
                analytic_lines_dens[i].set_data(r_analytic, analytic_dens)
                analytic_lines_pres[i].set_data(r_analytic, analytic_pres)
            else:
                analytic_lines_dens[i].set_data([], [])
                analytic_lines_pres[i].set_data([], [])

            artists.extend([lines_dens[i], lines_pres[i], lines_vel[i], lines_energy[i],
                            analytic_lines_dens[i], analytic_lines_pres[i]])
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    # Adjust the data directory path as needed.
    data_dirs = [
        "/Users/guo/OSS/sphcode/sample/lane_emden/results/GDISPH/lane_emden/3D",
    ]
    plot_titles = ["Dataset 1"]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_3d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles)


if __name__ == "__main__":
    main()
