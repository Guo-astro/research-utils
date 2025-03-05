from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from common import load_dataframes_3d

def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Animate 3D scatter plots with separate 2D intersection plots (for Density, Pressure, and Velocity).
    Top row: 3D scatter (pos_x, pos_y, pos_z, colored by physics_key).
    Bottom three rows: Intersection cross-sections (points with pos_y near the median).
    """
    n_dirs = len(list_of_dataframes)
    fig = plt.figure(figsize=(5 * n_dirs, 12))
    axes_3d = []
    # Prepare lists for simulation and analytic lines in the 2D intersection plots.
    lines_dens = []
    lines_pres = []
    lines_vel = []
    analytic_lines_dens = []
    analytic_lines_pres = []
    analytic_lines_vel = []

    # Create first row: 3D scatter subplots.
    for i, dataframes in enumerate(list_of_dataframes):
        ax3d = fig.add_subplot(4, n_dirs, i + 1, projection="3d")
        axes_3d.append(ax3d)
        # Set 3D axis limits based on concatenated data.
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
        # Store the scatter object for updating later.
        ax3d._scatter_obj = scat

    # Create 2D intersection subplots for Density, Pressure, and Velocity (rows 2-4).
    axes_intersect = {}
    variables = [("dens", "Density"), ("pres", "Pressure"), ("vel", "Velocity")]
    # Mapping each variable to its row (row 2 for density, row 3 for pressure, row 4 for velocity)
    for var, ylabel in variables:
        axes_intersect[var] = []
        row = {"dens": 1, "pres": 2, "vel": 3}[var]
        for i in range(n_dirs):
            ax = fig.add_subplot(4, n_dirs, n_dirs * row + i + 1)
            ax.set_xlabel("pos_x [m]")
            ax.set_ylabel(ylabel)
            # Create simulation line with markersize set to 2.
            line_sim, = ax.plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
            # Create analytic solution line (currently left empty).
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
        # Clear the 3D scatter data.
        for ax in axes_3d:
            ax._scatter_obj._offsets3d = (np.array([]), np.array([]), np.array([]))
        # Clear all intersection line data.
        for line in (lines_dens + lines_pres + lines_vel +
                     analytic_lines_dens + analytic_lines_pres + analytic_lines_vel):
            line.set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    def update(frame_index):
        # Loop over each dataset/direction.
        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            # Update 3D scatter plot.
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scat = axes_3d[i]._scatter_obj
            scat._offsets3d = (x, y, z)
            scat.set_array(df[physics_key].values)

            # Select points near the median of pos_y for intersection.
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
                # Calculate velocity magnitude if available.
                if "vel_z" in df.columns:
                    vel_slice = np.sqrt(
                        df["vel_x"].values[mask]**2 +
                        df["vel_y"].values[mask]**2 +
                        df["vel_z"].values[mask]**2)[sorted_idx]
                else:
                    vel_slice = np.sqrt(
                        df["vel_x"].values[mask]**2 +
                        df["vel_y"].values[mask]**2)[sorted_idx]

                # Update simulation intersection lines.
                lines_dens[i].set_data(x_sorted, dens_slice)
                lines_pres[i].set_data(x_sorted, pres_slice)
                lines_vel[i].set_data(x_sorted, vel_slice)

                # Optionally, update analytic lines if an analytic solution is known.
                # (For now, these are left empty.)
                for ax, data in zip([fig.axes[n_dirs + i], fig.axes[2 * n_dirs + i], fig.axes[3 * n_dirs + i]],
                                    [dens_slice, pres_slice, vel_slice]):
                    ax.set_xlim(x_sorted.min(), x_sorted.max())
                    ax.set_ylim(data.min(), data.max())
            else:
                # If no points are near the median, clear the intersection lines.
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
    # Update the data directory path as needed.
    data_dirs = [
        "/Users/guo/OSS/sphcode/sample/lane_emden/results/GSPH/lane_emden/3D",
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
