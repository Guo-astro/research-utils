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
    Subplots in each row share the same x and y limits, although different rows can have different ranges.
    """
    n_dirs = len(list_of_dataframes)

    # ----- Precompute Global Limits for Each Row -----
    # Row 0 (Scatter): use all pos_x and pos_y values from all datasets and frames.
    scatter_pos_x = []
    scatter_pos_y = []
    for dataframes in list_of_dataframes:
        for df in dataframes:
            scatter_pos_x.append(df["pos_x"].values)
            scatter_pos_y.append(df["pos_y"].values)
    scatter_pos_x = np.concatenate(scatter_pos_x)
    scatter_pos_y = np.concatenate(scatter_pos_y)
    scatter_x_lim = (scatter_pos_x.min() * 0.95, scatter_pos_x.max() * 1.05)
    scatter_y_lim = (scatter_pos_y.min() * 0.95, scatter_pos_y.max() * 1.05)

    # Rows 1-3 (Intersections): gather all slice data.
    density_x, density_y = [], []
    pressure_x, pressure_y = [], []
    velocity_x, velocity_y = [], []

    for dataframes in list_of_dataframes:
        for df in dataframes:
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol = 0.1 * (pos_y.max() - pos_y.min())
            mask = np.abs(pos_y - median_y) < tol
            if np.any(mask):
                x_slice = df["pos_x"].values[mask]
                # sort by x for consistency
                sorted_idx = np.argsort(x_slice)
                x_sorted = x_slice[sorted_idx]
                dens_slice = df["dens"].values[mask][sorted_idx]
                pres_slice = df["pres"].values[mask][sorted_idx]
                vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)[sorted_idx]

                density_x.append(x_sorted)
                density_y.append(dens_slice)
                pressure_x.append(x_sorted)
                pressure_y.append(pres_slice)
                velocity_x.append(x_sorted)
                velocity_y.append(vel_slice)

    if density_x:
        density_x_all = np.concatenate(density_x)
        density_y_all = np.concatenate(density_y)
        dens_x_lim = (density_x_all.min(), density_x_all.max())
        dens_y_lim = (density_y_all.min(), density_y_all.max())
    else:
        dens_x_lim, dens_y_lim = (0, 1), (0, 1)

    if pressure_x:
        pressure_x_all = np.concatenate(pressure_x)
        pressure_y_all = np.concatenate(pressure_y)
        pres_x_lim = (pressure_x_all.min(), pressure_x_all.max())
        pres_y_lim = (pressure_y_all.min(), pressure_y_all.max())
    else:
        pres_x_lim, pres_y_lim = (0, 1), (0, 1)

    if velocity_x:
        velocity_x_all = np.concatenate(velocity_x)
        velocity_y_all = np.concatenate(velocity_y)
        vel_x_lim = (velocity_x_all.min(), velocity_x_all.max())
        vel_y_lim = (velocity_y_all.min(), velocity_y_all.max())
    else:
        vel_x_lim, vel_y_lim = (0, 1), (0, 1)

    # ----- Create Figure and Subplots -----
    fig, axes = plt.subplots(4, n_dirs, figsize=(5 * n_dirs, 10))

    # Top row: Scatter plots with global scatter limits.
    scatters = []
    for i, dataframes in enumerate(list_of_dataframes):
        ax = axes[0, i]
        ax.set_xlim(scatter_x_lim)
        ax.set_ylim(scatter_y_lim)
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("pos_y [m]")
        # Use the physics key for color scaling based on all data in this dataset.
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax.scatter([], [], s=30, c=[], cmap="viridis",
                          vmin=physics_all.min(), vmax=physics_all.max())
        scatters.append(scat)
        ax.set_title(plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")
        add_sph_formulas(ax)

    # Intersection rows: Each row uses its precomputed global limits.
    lines_dens = []
    lines_pres = []
    lines_vel = []

    # Row 1: Density intersections.
    for i in range(n_dirs):
        ax = axes[1, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Density")
        ax.set_xlim(dens_x_lim)
        ax.set_ylim(dens_y_lim)
        line, = ax.plot([], [], marker='o', linestyle='-')
        lines_dens.append(line)

    # Row 2: Pressure intersections.
    for i in range(n_dirs):
        ax = axes[2, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Pressure")
        ax.set_xlim(pres_x_lim)
        ax.set_ylim(pres_y_lim)
        line, = ax.plot([], [], marker='o', linestyle='-')
        lines_pres.append(line)

    # Row 3: Velocity intersections.
    for i in range(n_dirs):
        ax = axes[3, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Velocity")
        ax.set_xlim(vel_x_lim)
        ax.set_ylim(vel_y_lim)
        line, = ax.plot([], [], marker='o', linestyle='-')
        lines_vel.append(line)

    n_frames = len(list_of_dataframes[0])

    # ----- Animation Functions -----
    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        for line in lines_dens + lines_pres + lines_vel:
            line.set_data([], [])
        return scatters + lines_dens + lines_pres + lines_vel

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            # Update scatter plot in row 0.
            x = df["pos_x"].values
            y = df["pos_y"].values
            colors = df[physics_key].values
            scatters[i].set_offsets(np.column_stack((x, y)))
            scatters[i].set_array(colors)

            # Intersection: choose points near the median of pos_y.
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

                lines_dens[i].set_data(x_sorted, dens_slice)
                lines_pres[i].set_data(x_sorted, pres_slice)
                lines_vel[i].set_data(x_sorted, vel_slice)
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
        "/Users/guo/research/sim_result_vis/result_data/khi_gsph/results"
    ]
    plot_titles = ["Dataset 1", "Dataset 2", "GSPH"]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_2d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles)


if __name__ == "__main__":
    main()
