# plot3d.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from common import load_dataframes_3d


def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Animate 3D scatter plots with separate 2D intersection plots (for Density, Pressure, and Velocity).
    Top row: 3D scatter (pos_x, pos_y, pos_z, colored by physics_key).
    Bottom three rows: Intersection cross-sections (points with pos_y near the median)
             for Density, Pressure, and Velocity.
    """
    n_dirs = len(list_of_dataframes)
    # Create a figure with 4 rows and n_dirs columns.
    fig = plt.figure(figsize=(5 * n_dirs, 12))
    axes_3d = []
    # Pre-create lists for the intersection line objects.
    lines_dens = []
    lines_pres = []
    lines_vel = []
    # Add subplots: first row for 3D scatter.
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
        ax3d.set_title(plot_titles[i] + " (3D)" if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")
        # Store the scatter in the axes for later update.
        ax3d._scatter_obj = scat
    # Bottom three rows: intersections (2D)
    axes_intersect = {}
    for var, row, ylabel in zip(["dens", "pres", "vel"], [1, 2, 3], ["Density", "Pressure", "Velocity"]):
        axes_intersect[var] = []
        for i in range(n_dirs):
            ax = fig.add_subplot(4, n_dirs, n_dirs * row + i + 1)
            ax.set_xlabel("pos_x [m]")
            ax.set_ylabel(ylabel)
            line, = ax.plot([], [], marker='o', linestyle='-')
            axes_intersect[var].append(ax)
            # Store the line object in a dictionary list.
            if var == "dens":
                lines_dens.append(line)
            elif var == "pres":
                lines_pres.append(line)
            elif var == "vel":
                lines_vel.append(line)
            ax.set_title((plot_titles[i] + f" {ylabel} Intersection") if plot_titles and i < len(
                plot_titles) else f"Dir {i + 1} {ylabel}")
    n_frames = len(list_of_dataframes[0])

    def init():
        for i, dataframes in enumerate(list_of_dataframes):
            ax3d = fig.axes[i]
            ax3d._scatter_obj._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line in lines_dens + lines_pres + lines_vel:
            line.set_data([], [])
        return [ax._scatter_obj for ax in axes_3d] + lines_dens + lines_pres + lines_vel

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
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
                for ax, data in zip([fig.axes[n_dirs + i], fig.axes[2 * n_dirs + i], fig.axes[3 * n_dirs + i]],
                                    [dens_slice, pres_slice, vel_slice]):
                    ax.set_xlim(x_sorted.min(), x_sorted.max())
                    ax.set_ylim(data.min(), data.max())
            else:
                lines_dens[i].set_data([], [])
                lines_pres[i].set_data([], [])
                lines_vel[i].set_data([], [])
        return [ax._scatter_obj for ax in axes_3d] + lines_dens + lines_pres + lines_vel

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    # Example usage for 3D plots.
    data_dirs = [
        "/Users/guo/OSS/sphcode/results",
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
