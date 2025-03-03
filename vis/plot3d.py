# plot3d_2p5d.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from common import load_dataframes_3d
from sod_analytic.shocktube_analytic_2p5d import analytic_sod_solution_2p5d


def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None,
                        dz_sim=1.0, h=0.01):
    """
    Animate 3D scatter plots with separate 2D intersection plots (for Density, Pressure, and Velocity)
    and overlay the analytic shock-tube solution (rescaled for 2.5D).

    Top row: 3D scatter (pos_x, pos_y, pos_z, colored by physics_key).
    Bottom three rows: Intersection cross-sections (points with pos_y near the median)
             for Density, Pressure, and Velocity, with the analytic solution overlaid.

    Parameters:
      dz_sim: effective layer thickness used in the simulation.
      h: smoothing length used in the cubic spline kernel.

    The analytic solution is scaled by F = dz_sim * (0.70/h) to account for the 3D kernel integration.
    """
    n_dirs = len(list_of_dataframes)
    fig = plt.figure(figsize=(5 * n_dirs, 12))
    axes_3d = []
    # Create lists for simulation intersection line objects and analytic line objects.
    lines_dens = []
    lines_pres = []
    lines_vel = []
    analytic_lines_dens = []
    analytic_lines_pres = []
    analytic_lines_vel = []

    # First row: 3D scatter plots.
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
        scat = ax3d.scatter([], [], [], s=3, c=[], cmap="viridis",
                            vmin=physics_all.min(), vmax=physics_all.max())
        title = (plot_titles[i] + " (3D)") if (plot_titles and i < len(plot_titles)) else f"Dir {i + 1}"
        ax3d.set_title(title)
        ax3d._scatter_obj = scat

    # Bottom three rows: intersections (2D) for Density, Pressure, and Velocity.
    axes_intersect = {}
    for var, row, ylabel in zip(["dens", "pres", "vel"], [1, 2, 3], ["Density", "Pressure", "Velocity"]):
        axes_intersect[var] = []
        for i in range(n_dirs):
            ax = fig.add_subplot(4, n_dirs, n_dirs * row + i + 1)
            ax.set_xlabel("pos_x [m]")
            ax.set_ylabel(ylabel)
            # Simulation line (solid line with markers)
            line_sim, = ax.plot([], [], marker='o', linestyle='-', label='Sim')
            # Analytic solution line (dashed red line)
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
                analytic_lines_vel.append(line_analytic)
            ax.legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        for i, dataframes in enumerate(list_of_dataframes):
            ax3d = fig.axes[i]
            ax3d._scatter_obj._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line in (lines_dens + lines_pres + lines_vel +
                     analytic_lines_dens + analytic_lines_pres + analytic_lines_vel):
            line.set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    def update(frame_index):
        # Determine the scaling factor to apply to the analytic solution:
        # scaling_factor = dz_sim * (0.70/h)
        scaling_factor = dz_sim * (0.70 / h)
        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            t = times[frame_index]
            # Update 3D scatter.
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scat = fig.axes[i]._scatter_obj
            scat._offsets3d = (x, y, z)
            scat.set_array(df[physics_key].values)

            # Intersection: select points with pos_y near the median.
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol = 0.1 * (pos_y.max() - pos_y.min())
            mask = np.abs(pos_y - median_y) < tol
            if np.any(mask):
                x_slice = df["pos_x"].values[mask]
                sorted_idx = np.argsort(x_slice)
                x_sorted = x_slice[sorted_idx]
                dens_slice = df["dens"].values[mask][sorted_idx]
                pres_slice = df["pres"].values[mask][sorted_idx]
                if "vel_z" in df.columns:
                    vel_slice = np.sqrt(
                        df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2 + df["vel_z"].values[mask] ** 2)[
                        sorted_idx]
                else:
                    vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)[sorted_idx]
                lines_dens[i].set_data(x_sorted, dens_slice)
                lines_pres[i].set_data(x_sorted, pres_slice)
                lines_vel[i].set_data(x_sorted, vel_slice)

                # Compute analytic solution on a uniform grid spanning the slice.
                x_grid = np.linspace(x_sorted.min(), x_sorted.max(), 500)
                # Call the analytic solution function.
                # It returns the 1D solution scaled by the factor scaling_factor.
                h_left = 0.70 / 170  # e.g., from left region mass/density relation
                h_right = 0.70 / 20  # e.g., from right region mass/density relation
                rho_analytic, pres_analytic, u_analytic = analytic_sod_solution_2p5d(
                    x_grid, t, gamma=1.4, x0=0.0, dz=1.0, h_right=h_right, h_left=h_left)
                analytic_lines_dens[i].set_data(x_grid, rho_analytic)
                analytic_lines_pres[i].set_data(x_grid, pres_analytic)
                analytic_lines_vel[i].set_data(x_grid, u_analytic)

                # Adjust axis limits.
                for ax, data in zip([fig.axes[n_dirs + i], fig.axes[2 * n_dirs + i], fig.axes[3 * n_dirs + i]],
                                    [dens_slice, pres_slice, vel_slice]):
                    ax.set_xlim(x_sorted.min(), x_sorted.max())
                    ax.set_ylim(data.min(), data.max())
            else:
                lines_dens[i].set_data([], [])
                lines_pres[i].set_data([], [])
                lines_vel[i].set_data([], [])
                analytic_lines_dens[i].set_data([], [])
                analytic_lines_pres[i].set_data([], [])
                analytic_lines_vel[i].set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    # Example usage for 2.5D plots.
    data_dirs = [
        "/Users/guo/OSS/sphcode/results",  # Update with your actual results directory
    ]
    plot_titles = ["Dataset 1"]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_3d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles,
                        dz_sim=1.0, h=0.05)


if __name__ == "__main__":
    main()
