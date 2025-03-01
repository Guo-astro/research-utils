# plot2d.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from common import add_sph_formulas, load_dataframes_2d


def animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Animate 2D scatter plots with separate intersection rows for Density, Pressure, and Velocity.
    Top row: original scatter (pos_x vs pos_y, colored by physics_key).
    Rows 2-4: Intersection cross-section (using points where pos_y is near its median)
             for Density, Pressure, and Velocity (computed from vel_x, vel_y).
    The y-axis for all intersection plots is fixed based on the overall min/max values.
    """
    n_dirs = len(list_of_dataframes)
    # Create a figure with 4 rows and n_dirs columns.
    fig, axes = plt.subplots(4, n_dirs, figsize=(5 * n_dirs, 10), sharex='col')
    # Top row: scatter plots
    scatters = []
    for i, dataframes in enumerate(list_of_dataframes):
        ax = axes[0, i]
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
        ax.set_xlim(pos_x_all.min() * 0.95, pos_x_all.max() * 1.05)
        ax.set_ylim(pos_y_all.min() * 0.95, pos_y_all.max() * 1.05)
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("pos_y [m]")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax.scatter([], [], s=30, c=[], cmap="viridis",
                          vmin=physics_all.min(), vmax=physics_all.max())
        scatters.append(scat)
        ax.set_title(plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")
        # Optionally add SPH formulas if desired:
        add_sph_formulas(ax)

    # Precompute fixed y-axis limits for the intersection plots for each dataset.
    fixed_limits = []
    for i, dataframes in enumerate(list_of_dataframes):
        dens_values = []
        pres_values = []
        vel_values = []
        for df in dataframes:
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol = 0.1 * (pos_y.max() - pos_y.min())
            mask = np.abs(pos_y - median_y) < tol
            if np.any(mask):
                dens_values.append(df["dens"].values[mask])
                pres_values.append(df["pres"].values[mask])
                vel = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)
                vel_values.append(vel)
        # Combine all values across frames if available, otherwise default to (0,1)
        if dens_values:
            dens_all = np.concatenate(dens_values)
            dens_lim = (dens_all.min(), dens_all.max())
        else:
            dens_lim = (0, 1)
        if pres_values:
            pres_all = np.concatenate(pres_values)
            pres_lim = (pres_all.min(), pres_all.max())
        else:
            pres_lim = (0, 1)
        if vel_values:
            vel_all = np.concatenate(vel_values)
            vel_lim = (vel_all.min(), vel_all.max())
        else:
            vel_lim = (0, 1)
        fixed_limits.append({"dens": dens_lim, "pres": pres_lim, "vel": vel_lim})

    # Bottom three rows: intersections for Density, Pressure, Velocity
    # Pre-create line objects in each intersection subplot.
    lines_dens = []
    lines_pres = []
    lines_vel = []
    for i in range(n_dirs):
        for row, lines_list, label in zip([1, 2, 3],
                                          [lines_dens, lines_pres, lines_vel],
                                          ["Density", "Pressure", "Velocity"]):
            ax = axes[row, i]
            ax.set_xlabel("pos_x [m]")
            ax.set_ylabel(label)
            # Set the fixed y-axis limits here
            if label == "Density":
                ax.set_ylim(fixed_limits[i]["dens"])
            elif label == "Pressure":
                ax.set_ylim(fixed_limits[i]["pres"])
            elif label == "Velocity":
                ax.set_ylim(fixed_limits[i]["vel"])
            line, = ax.plot([], [], marker='o', linestyle='-')
            lines_list.append(line)

    n_frames = len(list_of_dataframes[0])

    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        for line in lines_dens + lines_pres + lines_vel:
            line.set_data([], [])
        return scatters + lines_dens + lines_pres + lines_vel

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            # Update scatter plot
            x = df["pos_x"].values
            y = df["pos_y"].values
            colors = df[physics_key].values
            scatters[i].set_offsets(np.column_stack((x, y)))
            scatters[i].set_array(colors)
            # Intersection: use points with pos_y near the median.
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
                vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)[sorted_idx]
                # Update lines with current frame data.
                lines_dens[i].set_data(x_sorted, dens_slice)
                lines_pres[i].set_data(x_sorted, pres_slice)
                lines_vel[i].set_data(x_sorted, vel_slice)
                # For x-axis we update dynamically; for y-axis we use fixed limits.
                axes[1, i].set_xlim(x_sorted.min(), x_sorted.max())
                axes[2, i].set_xlim(x_sorted.min(), x_sorted.max())
                axes[3, i].set_xlim(x_sorted.min(), x_sorted.max())
            else:
                lines_dens[i].set_data([], [])
                lines_pres[i].set_data([], [])
                lines_vel[i].set_data([], [])
        return scatters + lines_dens + lines_pres + lines_vel

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    # Example usage for 2D plots.
    data_dirs = [
        "/Users/guo/research/sim_result_vis/result_data/khi_disph/results",
        "/Users/guo/research/sim_result_vis/result_data/khi_ssph/results",
    ]
    plot_titles = ["Dataset 1", "Dataset 2"]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_2d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles)


if __name__ == "__main__":
    main()
