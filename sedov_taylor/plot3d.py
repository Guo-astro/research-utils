from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Animate 3D scatter plots with separate 2D intersection plots (for Density, Pressure, and Radial Velocity).
    Top row: 3D scatter (pos_x, pos_y, pos_z, colored by physics_key).
    Bottom three rows: Intersection cross-sections (radial distance from (0.5, 0.5, 0.5) vs. quantities).
    """
    n_dirs = len(list_of_dataframes)
    fig = plt.figure(figsize=(5 * n_dirs, 12))
    axes_3d = []
    lines_dens = []
    lines_pres = []
    lines_vel = []
    analytic_lines_dens = []
    analytic_lines_pres = []
    analytic_lines_vel = []

    # Create first row: 3D scatter subplots
    for i, dataframes in enumerate(list_of_dataframes):
        ax3d = fig.add_subplot(4, n_dirs, i + 1, projection="3d")
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

    # Create 2D intersection subplots for Density, Pressure, and Radial Velocity
    axes_intersect = {}
    variables = [("dens", "Density"), ("pres", "Pressure"), ("vel", "Radial Velocity")]
    for var, ylabel in variables:
        axes_intersect[var] = []
        row = {"dens": 1, "pres": 2, "vel": 3}[var]
        for i in range(n_dirs):
            ax = fig.add_subplot(4, n_dirs, n_dirs * row + i + 1)
            ax.set_xlabel("r [m]")  # Updated to radial distance
            ax.set_ylabel(ylabel)
            line_sim, = ax.plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
            line_analytic, = ax.plot([], [], linestyle='-', color='red', label='Analytic')  # Solid line
            axes_intersect[var].append(ax)
            if var == "dens":
                lines_dens.append(line_sim)
                analytic_lines_dens.append(line_analytic)
            elif var == "pres":
                lines_pres.append(line_sim)
                analytic_lines_pres.append(line_analytic)
            elif var == "vel":
                lines_vel.append(line_sim)
                analytic_lines_vel.append(line_analytic)
            ax.legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        for ax in axes_3d:
            ax._scatter_obj._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line in (lines_dens + lines_pres + lines_vel +
                     analytic_lines_dens + analytic_lines_pres + analytic_lines_vel):
            line.set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    def update(frame_index):
        center = np.array([0.5, 0.5, 0.5])
        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            # Update 3D scatter plot
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scat = axes_3d[i]._scatter_obj
            scat._offsets3d = (x, y, z)
            scat.set_array(df[physics_key].values)

            # Compute radial distance for all particles
            pos = df[["pos_x", "pos_y", "pos_z"]].values
            dr = pos - center
            r = np.sqrt(np.sum(dr**2, axis=1))

            # Select particles near y = 0.5 (center)
            pos_y = df["pos_y"].values
            tol = 0.1 * (pos_y.max() - pos_y.min())
            mask = np.abs(pos_y - 0.5) < tol
            if np.any(mask):
                r_slice = r[mask]
                dens_slice = df["dens"].values[mask]
                pres_slice = df["pres"].values[mask]
                vel = df[["vel_x", "vel_y", "vel_z"]].values[mask]
                dr_slice = dr[mask]
                # Compute radial velocity
                r_slice_nonzero = r_slice > 0
                v_rad_slice = np.zeros_like(r_slice)
                v_rad_slice[r_slice_nonzero] = np.sum(vel[r_slice_nonzero] * dr_slice[r_slice_nonzero], axis=1) / r_slice[r_slice_nonzero]

                # Sort by radial distance
                sorted_idx = np.argsort(r_slice)
                r_sorted = r_slice[sorted_idx]
                dens_sorted = dens_slice[sorted_idx]
                pres_sorted = pres_slice[sorted_idx]
                v_rad_sorted = v_rad_slice[sorted_idx]

                # Update simulation lines
                lines_dens[i].set_data(r_sorted, dens_sorted)
                lines_pres[i].set_data(r_sorted, pres_sorted)
                lines_vel[i].set_data(r_sorted, v_rad_sorted)

                # Placeholder for analytical solution (Sedov-Taylor)
                r_analytic = np.linspace(0, r_sorted.max(), 100)
                t = times[frame_index]  # Current time
                # TODO: Replace with actual Sedov-Taylor solution
                # Example: r_shock = (E * t^2 / rho_0)^(1/5), E=1, rho_0=1
                dens_analytic = np.ones_like(r_analytic) * 1.0  # Background density
                pres_analytic = np.zeros_like(r_analytic)
                v_rad_analytic = np.zeros_like(r_analytic)
                analytic_lines_dens[i].set_data(r_analytic, dens_analytic)
                analytic_lines_pres[i].set_data(r_analytic, pres_analytic)
                analytic_lines_vel[i].set_data(r_analytic, v_rad_analytic)

                # Adjust axis limits
                for ax, data in zip([axes_intersect["dens"][i], axes_intersect["pres"][i], axes_intersect["vel"][i]],
                                    [dens_sorted, pres_sorted, v_rad_sorted]):
                    ax.set_xlim(r_sorted.min(), r_sorted.max())
                    ax.set_ylim(data.min(), data.max())
            else:
                # Clear lines if no particles are selected
                for lines in [lines_dens, lines_pres, lines_vel,
                              analytic_lines_dens, analytic_lines_pres, analytic_lines_vel]:
                    lines[i].set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def main():
    data_dirs = [
        "/Users/guo/OSS/sphcode/sample/sedov_taylor/results/GSPH/sedov_taylor/3D",
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