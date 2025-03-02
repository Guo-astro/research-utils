# plot2d.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from vis.common import load_dataframes_2d


# --- Analytic functions for x-slice (perfect match from initial condition) ---
def analytic_density_profile(x):
    """Analytic density profile for x-slice: perfect match (central region = 2)."""
    return 2.0 * np.ones_like(x)


def analytic_pressure_profile(x):
    """Analytic pressure profile for x-slice: constant 2.5."""
    return 2.5 * np.ones_like(x)


def analytic_velocity_profile(x):
    """Analytic velocity profile for x-slice: constant 0.5 (from region 2)."""
    return 0.5 * np.ones_like(x)


# --- New Analytic functions for y-slice (slice in x) ---
def analytic_density_profile_y(y):
    """
    Perfect match: density = 2 if 0.25 < y < 0.75, else 1.
    """
    return np.where((y > 0.25) & (y < 0.75), 2.0, 1.0)


def analytic_pressure_profile_y(y):
    """Perfect match: pressure = 2.5 everywhere."""
    return 2.5 * np.ones_like(y)


def analytic_velocity_profile_y(y):
    """
    Perfect match: horizontal velocity = 0.5 if 0.25 < y < 0.75, else -0.5.
    (We use the horizontal component as the analytic velocity for the y-slice.)
    """
    return np.where((y > 0.25) & (y < 0.75), 0.5, -0.5)


def animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Create an animation with 7 rows per dataset:
      Row 0: Scatter plot (pos_x vs pos_y)
      Rows 1-3: Intersection slices using points with pos_y near its median
                (x vs density, pressure, velocity)
      Rows 4-6: Intersection slices using points with pos_x near its median
                (y vs density, pressure, velocity)
    Analytic overlays (computed from the initial condition) are added to each intersection subplot.
    """
    n_dirs = len(list_of_dataframes)

    ### Precompute global limits for the scatter (row 0) and for x-slice rows (rows 1-3)
    # Row 0 (Scatter): use all pos_x and pos_y values.
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

    # Rows 1-3 (x-slice): use points where pos_y is near the median.
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
                sorted_idx = np.argsort(x_slice)
                x_sorted = x_slice[sorted_idx]
                density_x.append(x_sorted)
                density_y.append(df["dens"].values[mask][sorted_idx])
                pressure_x.append(x_sorted)
                pressure_y.append(df["pres"].values[mask][sorted_idx])
                velocity_x.append(x_sorted)
                vel_sorted = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)[sorted_idx]
                velocity_y.append(vel_sorted)

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

    ### Precompute global limits for the y-slice rows (rows 4-6)
    # Here, we select points where pos_x is near its median.
    dens_y_ind_list, dens_y_prop_list = [], []
    pres_y_ind_list, pres_y_prop_list = [], []
    vel_y_ind_list, vel_y_prop_list = [], []
    for dataframes in list_of_dataframes:
        for df in dataframes:
            pos_x = df["pos_x"].values
            median_x = np.median(pos_x)
            tol = 0.1 * (pos_x.max() - pos_x.min())
            mask = np.abs(pos_x - median_x) < tol
            if np.any(mask):
                y_slice = df["pos_y"].values[mask]
                sorted_idx = np.argsort(y_slice)
                y_sorted = y_slice[sorted_idx]
                dens_y_ind_list.append(y_sorted)
                dens_y_prop_list.append(df["dens"].values[mask][sorted_idx])
                pres_y_ind_list.append(y_sorted)
                pres_y_prop_list.append(df["pres"].values[mask][sorted_idx])
                vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)[sorted_idx]
                vel_y_ind_list.append(y_sorted)
                vel_y_prop_list.append(vel_slice)

    if dens_y_ind_list:
        dens_y_ind_all = np.concatenate(dens_y_ind_list)
        dens_y_prop_all = np.concatenate(dens_y_prop_list)
        y_dens_x_lim = (dens_y_ind_all.min(), dens_y_ind_all.max())
        y_dens_y_lim = (dens_y_prop_all.min(), dens_y_prop_all.max())
    else:
        y_dens_x_lim, y_dens_y_lim = (0, 1), (0, 1)

    if pres_y_ind_list:
        pres_y_ind_all = np.concatenate(pres_y_ind_list)
        pres_y_prop_all = np.concatenate(pres_y_prop_list)
        y_pres_x_lim = (pres_y_ind_all.min(), pres_y_ind_all.max())
        y_pres_y_lim = (pres_y_prop_all.min(), pres_y_prop_all.max())
    else:
        y_pres_x_lim, y_pres_y_lim = (0, 1), (0, 1)

    if vel_y_ind_list:
        vel_y_ind_all = np.concatenate(vel_y_ind_list)
        vel_y_prop_all = np.concatenate(vel_y_prop_list)
        y_vel_x_lim = (vel_y_ind_all.min(), vel_y_ind_all.max())
        y_vel_y_lim = (vel_y_prop_all.min(), vel_y_prop_all.max())
    else:
        y_vel_x_lim, y_vel_y_lim = (0, 1), (0, 1)

    ### Create figure with 7 rows and n_dirs columns.
    fig, axes = plt.subplots(7, n_dirs, figsize=(5 * n_dirs, 12))
    if n_dirs == 1:
        axes = np.expand_dims(axes, axis=1)

    # --- Row 0: Scatter plots (pos_x vs pos_y) ---
    scatters = []
    for i, dataframes in enumerate(list_of_dataframes):
        ax = axes[0, i]
        ax.set_xlim(scatter_x_lim)
        ax.set_ylim(scatter_y_lim)
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("pos_y [m]")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax.scatter([], [], s=30, c=[], cmap="viridis",
                          vmin=physics_all.min(), vmax=physics_all.max())
        scatters.append(scat)
        ax.set_title(plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")

    # --- Rows 1-3: x-slice plots (independent variable = pos_x) ---
    # Row 1: Density vs pos_x (with analytic overlay)
    sim_dens_lines = []
    for i in range(n_dirs):
        ax = axes[1, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Density")
        ax.set_xlim(dens_x_lim)
        ax.set_ylim(dens_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Density")
        sim_dens_lines.append(sim_line)
        ax.legend(loc="upper right")

    # Row 2: Pressure vs pos_x (with analytic overlay)
    sim_pres_lines = []
    analytic_pres_lines = []
    for i in range(n_dirs):
        ax = axes[2, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Pressure")
        ax.set_xlim(pres_x_lim)
        ax.set_ylim(pres_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Pressure")
        sim_pres_lines.append(sim_line)
        ax.legend(loc="upper right")

    # Row 3: Velocity vs pos_x (with analytic overlay)
    sim_vel_lines = []
    analytic_vel_lines = []
    for i in range(n_dirs):
        ax = axes[3, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Velocity")
        ax.set_xlim(vel_x_lim)
        ax.set_ylim(vel_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Velocity")
        sim_vel_lines.append(sim_line)
        ax.legend(loc="upper right")

    # --- Rows 4-6: y-slice plots (independent variable = pos_y) ---
    # Row 4: Density vs pos_y (with analytic overlay)
    sim_dens_y_lines = []
    analytic_dens_y_lines = []
    for i in range(n_dirs):
        ax = axes[4, i]
        ax.set_xlabel("pos_y [m]")
        ax.set_ylabel("Density")
        ax.set_xlim(y_dens_x_lim)
        ax.set_ylim(y_dens_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Density")
        sim_dens_y_lines.append(sim_line)
        ax.legend(loc="upper right")

    # Row 5: Pressure vs pos_y (with analytic overlay)
    sim_pres_y_lines = []
    analytic_pres_y_lines = []
    for i in range(n_dirs):
        ax = axes[5, i]
        ax.set_xlabel("pos_y [m]")
        ax.set_ylabel("Pressure")
        ax.set_xlim(y_pres_x_lim)
        ax.set_ylim(y_pres_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Pressure")
        sim_pres_y_lines.append(sim_line)
        ax.legend(loc="upper right")

    # Row 6: Velocity vs pos_y (with analytic overlay)
    sim_vel_y_lines = []
    analytic_vel_y_lines = []
    for i in range(n_dirs):
        ax = axes[6, i]
        ax.set_xlabel("pos_y [m]")
        ax.set_ylabel("Velocity")
        ax.set_xlim(y_vel_x_lim)
        ax.set_ylim(y_vel_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Velocity")
        sim_vel_y_lines.append(sim_line)
        ax.legend(loc="upper right")

    n_frames = len(list_of_dataframes[0])

    # --- Initialize Analytic Overlays Once (using t = 0) ---
    # For x-slice (rows 1-3)
    x_dense_dens = np.linspace(dens_x_lim[0], dens_x_lim[1], 200)
    x_dense_pres = np.linspace(pres_x_lim[0], pres_x_lim[1], 200)
    x_dense_vel = np.linspace(vel_x_lim[0], vel_x_lim[1], 200)
    # For y-slice (rows 4-6)
    y_dense_dens = np.linspace(y_dens_x_lim[0], y_dens_x_lim[1], 200)
    y_dense_pres = np.linspace(y_pres_x_lim[0], y_pres_x_lim[1], 200)
    y_dense_vel = np.linspace(y_vel_x_lim[0], y_vel_x_lim[1], 200)

    def init():
        # Clear scatter plots.
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        # Clear simulation lines for x-slice rows.
        for line in sim_dens_lines + sim_pres_lines + sim_vel_lines:
            line.set_data([], [])
        # Set analytic overlays for x-slice rows.
        # Clear simulation lines for y-slice rows.
        for line in sim_dens_y_lines + sim_pres_y_lines + sim_vel_y_lines:
            line.set_data([], [])
        # Set analytic overlays for y-slice rows.
        return (scatters + analytic_dens_lines + analytic_pres_lines + analytic_vel_lines +
                sim_dens_lines + sim_pres_lines + sim_vel_lines +
                analytic_dens_y_lines + analytic_pres_y_lines + analytic_vel_y_lines +
                sim_dens_y_lines + sim_pres_y_lines + sim_vel_y_lines)

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            # --- Update scatter plot (row 0) ---
            x = df["pos_x"].values
            y = df["pos_y"].values
            colors = df[physics_key].values
            scatters[i].set_offsets(np.column_stack((x, y)))
            scatters[i].set_array(colors)

            # --- Update x-slice curves (rows 1-3) ---
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol_y = 0.1 * (pos_y.max() - pos_y.min())
            mask_y = np.abs(pos_y - median_y) < tol_y
            if np.any(mask_y):
                x_slice = df["pos_x"].values[mask_y]
                sorted_idx = np.argsort(x_slice)
                x_sorted = x_slice[sorted_idx]
                dens_slice = df["dens"].values[mask_y][sorted_idx]
                pres_slice = df["pres"].values[mask_y][sorted_idx]
                vel_slice = np.sqrt(df["vel_x"].values[mask_y] ** 2 + df["vel_y"].values[mask_y] ** 2)[sorted_idx]
                sim_dens_lines[i].set_data(x_sorted, dens_slice)
                sim_pres_lines[i].set_data(x_sorted, pres_slice)
                sim_vel_lines[i].set_data(x_sorted, vel_slice)
            else:
                sim_dens_lines[i].set_data([], [])
                sim_pres_lines[i].set_data([], [])
                sim_vel_lines[i].set_data([], [])

            # --- Update y-slice curves (rows 4-6) ---
            pos_x = df["pos_x"].values
            median_x = np.median(pos_x)
            tol_x = 0.1 * (pos_x.max() - pos_x.min())
            mask_x = np.abs(pos_x - median_x) < tol_x
            if np.any(mask_x):
                y_slice = df["pos_y"].values[mask_x]
                sorted_idx = np.argsort(y_slice)
                y_sorted = y_slice[sorted_idx]
                dens_slice_y = df["dens"].values[mask_x][sorted_idx]
                pres_slice_y = df["pres"].values[mask_x][sorted_idx]
                vel_slice_y = np.sqrt(df["vel_x"].values[mask_x] ** 2 + df["vel_y"].values[mask_x] ** 2)[sorted_idx]
                sim_dens_y_lines[i].set_data(y_sorted, dens_slice_y)
                sim_pres_y_lines[i].set_data(y_sorted, pres_slice_y)
                sim_vel_y_lines[i].set_data(y_sorted, vel_slice_y)
            else:
                sim_dens_y_lines[i].set_data([], [])
                sim_pres_y_lines[i].set_data([], [])
                sim_vel_y_lines[i].set_data([], [])
        return (scatters + analytic_pres_lines + analytic_vel_lines +
                sim_dens_lines + sim_pres_lines + sim_vel_lines +
                analytic_dens_y_lines + analytic_pres_y_lines + analytic_vel_y_lines +
                sim_dens_y_lines + sim_pres_y_lines + sim_vel_y_lines)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    # Example usage for 2D plots.
    data_dirs = [
        "/Users/guo/OSS/sphcode/results",
    ]
    plot_titles = ["DISPH"]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_2d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles)


if __name__ == "__main__":
    main()
