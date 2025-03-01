import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

DEBUG_THRESHOLD = 1e5  # adjust this threshold as needed

# Define file patterns to exclude (e.g., files with "energy" in the name or ending with ".log")
EXCLUDE_PATTERNS = ["energy", ".log"]


def extract_time(file):
    """
    Extract the time step (in seconds) from the file header.
    Looks for a line like: "# Time [s]: 0"
    """
    with open(file, "r") as f:
        for line in f:
            if line.startswith("# Time"):
                match = re.search(r":\s*([\d\.\-eE]+)", line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
    return None


def filter_files(file_list):
    """
    Exclude files whose basenames contain any of the EXCLUDE_PATTERNS.
    """
    filtered = []
    for f in file_list:
        basename = os.path.basename(f)
        if not any(pattern in basename for pattern in EXCLUDE_PATTERNS):
            filtered.append(f)
        else:
            print(f"Excluding file: {basename}")
    return filtered


def load_dataframes(data_path, file_extension="*.dat"):
    """
    Load 2D data files from the specified directory into a list of DataFrames,
    while extracting the corresponding time steps from the header.
    Returns: (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 2D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 2D files in {data_path}.")

    times = [extract_time(file) for file in file_list]

    # Expected columns for 2D: (15 columns)
    columns = [
        "pos_x", "pos_y",
        "vel_x", "vel_y",
        "acc_x", "acc_y",
        "mass", "dens",
        "pres", "ene",
        "sml", "id",
        "neighbor", "alpha",
        "gradh"
    ]
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file, sep='\s+', comment='#', header=None, names=columns)
        # Debug: Check for unusually large pos_x values.
        large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
        if large_indices:
            print(f"DEBUG: In file '{file}', found large pos_x values at rows {large_indices}:")
            for idx in large_indices:
                print(f"  Row {idx}: pos_x = {df.loc[idx, 'pos_x']}")
        dataframes.append(df)
    return dataframes, times


def load_dataframes_3d(data_path, file_extension="*.dat"):
    """
    Load 3D data files from the specified directory into a list of DataFrames,
    while extracting the corresponding time steps from the header.
    Returns: (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 3D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 3D files in {data_path}.")

    times = [extract_time(file) for file in file_list]
    dataframes = []
    # Define expected column lists:
    basic_columns = [
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "acc_x", "acc_y", "acc_z",
        "mass", "dens", "pres", "ene",
        "sml", "id", "neighbor", "alpha", "gradh"
    ]
    columns_15 = [
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "mass", "dens", "pres", "ene",
        "sml", "id", "neighbor", "alpha", "gradh"
    ]
    for file in file_list:
        df = pd.read_csv(file, sep='\s+', comment='#', header=None)
        ncols = df.shape[1]
        if ncols == 33:
            additional_columns = [
                "grad_velocity_0_x", "grad_velocity_0_y", "grad_velocity_0_z",
                "grad_pressure_x", "grad_pressure_y", "grad_pressure_z",
                "grad_velocity_2_x", "grad_velocity_2_y", "grad_velocity_2_z",
                "grad_velocity_1_x", "grad_velocity_1_y", "grad_velocity_1_z",
                "grad_density_x", "grad_density_y", "grad_density_z"
            ]
            df.columns = basic_columns + additional_columns
        elif ncols == 18:
            df.columns = basic_columns
        elif ncols == 15:
            df.columns = columns_15
        else:
            # Fall back: assign the first ncols of basic_columns and warn.
            df.rename(columns=dict(zip(range(ncols), basic_columns[:ncols])), inplace=True)
            print(f"Warning: File '{file}' has {ncols} columns; expected 15, 18, or 33 columns.")
        # Optional debug check for large pos_x values
        large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
        if large_indices:
            print(f"DEBUG: In file '{file}', found large pos_x values at rows {large_indices}:")
            for idx in large_indices:
                print(f"  Row {idx}: pos_x = {df.loc[idx, 'pos_x']}")
        dataframes.append(df)
    return dataframes, times


def add_sph_formulas(ax):
    """
    Adds the SPH formulas to the axis.
    The standard SPH density estimation and the density-independent SPH pressure formulation are shown.
    """
    text = (
            r"Standard SPH: $\rho_i = \sum_j m_j\, W(|\mathbf{r}_i-\mathbf{r}_j|, h)$" "\n" +
            r"Density Ind. SPH: $P_i = P_0(\frac{\hat{\rho}_i}{\rho_0})^\gamma$, "
            r"$\hat{\rho}_i = \sum_j m_j (\frac{A_j}{A_i})^{1/\gamma} W(|\mathbf{r}_i-\mathbf{r}_j|, h)$"
    )
    ax.text(0.05, -.55, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7))


def animate_multiple_1d(list_of_dataframes, list_of_times, plot_titles=None):
    """
    Create one 1D scatter subplot per data directory.
    Each subplot shows pos_x vs. dens and is animated in sync.
    """
    n_dirs = len(list_of_dataframes)
    fig, axes = plt.subplots(1, n_dirs, figsize=(5 * n_dirs, 4))
    khi_formula = (r"$\omega = \frac{k(\rho_1 U_1 + \rho_2 U_2)}{\rho_1+\rho_2} \pm "
                   r"\sqrt{\frac{k^2 \rho_1 \rho_2 (U_1-U_2)^2}{(\rho_1+\rho_2)^2} - "
                   r"gk\frac{\rho_1-\rho_2}{\rho_1+\rho_2}}$")
    fig.suptitle("Kelvin–Helmholtz Instability:\n" + khi_formula, fontsize=12)

    if n_dirs == 1:
        axes = [axes]
    scatters = []
    for i, dataframes in enumerate(list_of_dataframes):
        ax = axes[i]
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        dens_all = np.concatenate([df["dens"].values for df in dataframes])
        ax.set_xlim(pos_x_all.min() - 0.05 * (pos_x_all.max() - pos_x_all.min()),
                    pos_x_all.max() + 0.05 * (pos_x_all.max() - pos_x_all.min()))
        ax.set_ylim(dens_all.min() - 0.05 * (dens_all.max() - dens_all.min()),
                    dens_all.max() + 0.05 * (dens_all.max() - dens_all.min()))
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("dens [kg/m³]")
        scat = ax.scatter([], [], s=10, c="blue")
        scatters.append(scat)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    n_frames = len(list_of_dataframes[0])

    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            x = df["pos_x"].values
            y = df["dens"].values
            scatters[i].set_offsets(np.column_stack((x, y)))
            base_title = plot_titles[i] if plot_titles is not None and i < len(plot_titles) else f"Dir {i + 1}"
            axes[i].set_title(base_title)
        return scatters

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=True)
    plt.show()
    return ani


def animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Create a 2D animated figure for each data directory that includes:
      - Top row: original 2D scatter plot (pos_x vs. pos_y, colored by physics_key).
      - Bottom three rows: Separate intersection plots for Density, Pressure, and Velocity.
        The intersection is taken as a cross‐section (points with pos_y near the median).
    """
    n_dirs = len(list_of_dataframes)
    # Create a figure with 4 rows (1 for scatter + 3 for intersections) and n_dirs columns.
    fig, axes = plt.subplots(4, n_dirs, figsize=(5 * n_dirs, 10))
    khi_formula = (r"$\omega = \frac{k(\rho_1 U_1 + \rho_2 U_2)}{\rho_1+\rho_2} \pm "
                   r"\sqrt{\frac{k^2 \rho_1 \rho_2 (U_1-U_2)^2}{(\rho_1+\rho_2)^2} - "
                   r"gk\frac{\rho_1-\rho_2}{\rho_1+\rho_2}}$")
    fig.suptitle("Kelvin–Helmholtz Instability:\n" + khi_formula, fontsize=12)

    # Ensure axes are 2D array even for n_dirs==1
    if n_dirs == 1:
        axes = np.expand_dims(axes, axis=1)

    # Top row: original 2D scatter plots
    scatters = []
    for i, dataframes in enumerate(list_of_dataframes):
        ax = axes[0, i]
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
        ax.set_xlim(pos_x_all.min() - 0.05 * (pos_x_all.max() - pos_x_all.min()),
                    pos_x_all.max() + 0.05 * (pos_x_all.max() - pos_x_all.min()))
        ax.set_ylim(pos_y_all.min() - 0.05 * (pos_y_all.max() - pos_y_all.min()),
                    pos_y_all.max() + 0.05 * (pos_y_all.max() - pos_y_all.min()))
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("pos_y [m]")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax.scatter([], [], s=30, c=[], cmap="viridis",
                          vmin=physics_all.min(), vmax=physics_all.max())
        scatters.append(scat)
        cbar = fig.colorbar(scat, ax=ax)
        cbar.set_label(f"{physics_key} value")
        ax.set_title(plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")

    # Bottom three rows: separate intersection plots for Density, Pressure, and Velocity.
    # Row indices: 1 = Density, 2 = Pressure, 3 = Velocity.
    intersection_lines = {'dens': [], 'pres': [], 'vel': []}
    for i in range(n_dirs):
        for var, row in zip(["dens", "pres", "vel"], [1, 2, 3]):
            ax = axes[row, i]
            ax.set_xlabel("pos_x [m]")
            if var == "dens":
                ax.set_ylabel("Density")
            elif var == "pres":
                ax.set_ylabel("Pressure")
            elif var == "vel":
                ax.set_ylabel("Velocity")
            line, = ax.plot([], [], marker='o', linestyle='-')
            intersection_lines[var].append(line)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    n_frames = len(list_of_dataframes[0])

    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        for var in intersection_lines:
            for line in intersection_lines[var]:
                line.set_data([], [])
        artists = scatters
        for var in intersection_lines:
            artists.extend(intersection_lines[var])
        return artists

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            # Update top row scatter plot
            x = df["pos_x"].values
            y = df["pos_y"].values
            colors = df[physics_key].values
            scatters[i].set_offsets(np.column_stack((x, y)))
            scatters[i].set_array(colors)

            # Intersection: select points with pos_y near the median
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol = 0.1 * (pos_y.max() - pos_y.min())
            mask = np.abs(pos_y - median_y) < tol
            x_slice = df["pos_x"].values[mask]
            if len(x_slice) > 0:
                sorted_indices = np.argsort(x_slice)
                x_sorted = x_slice[sorted_indices]
                # Density
                dens_slice = df["dens"].values[mask][sorted_indices]
                # Pressure
                pres_slice = df["pres"].values[mask][sorted_indices]
                # Velocity (2D: compute magnitude from vel_x, vel_y)
                vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)[sorted_indices]

                intersection_lines["dens"][i].set_data(x_sorted, dens_slice)
                intersection_lines["pres"][i].set_data(x_sorted, pres_slice)
                intersection_lines["vel"][i].set_data(x_sorted, vel_slice)

                # Adjust limits for each intersection subplot
                for var, data in zip(["dens", "pres", "vel"], [dens_slice, pres_slice, vel_slice]):
                    ax = axes[{"dens": 1, "pres": 2, "vel": 3}[var], i]
                    ax.set_xlim(x_sorted.min(), x_sorted.max())
                    ax.set_ylim(data.min(), data.max())
            else:
                for var in intersection_lines:
                    intersection_lines[var][i].set_data([], [])
        artists = scatters
        for var in intersection_lines:
            artists.extend(intersection_lines[var])
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.show()
    return ani


def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Create a 3D animated figure for each data directory that includes:
      - Top row: original 3D scatter plot (pos_x, pos_y, pos_z, colored by physics_key).
      - Bottom three rows: Separate 2D intersection plots for Density, Pressure, and Velocity.
        The intersection is taken as a cross‐section (points with pos_y near the median).
    """
    n_dirs = len(list_of_dataframes)
    # Create a figure with 4 rows and n_dirs columns.
    fig = plt.figure(figsize=(5 * n_dirs, 12))
    khi_formula = (r"$\omega = \frac{k(\rho_1 U_1 + \rho_2 U_2)}{\rho_1+\rho_2} \pm "
                   r"\sqrt{\frac{k^2 \rho_1 \rho_2 (U_1-U_2)^2}{(\rho_1+\rho_2)^2} - "
                   r"gk\frac{\rho_1-\rho_2}{\rho_1+\rho_2}}$")
    fig.suptitle("Kelvin–Helmholtz Instability:\n" + khi_formula, fontsize=12)

    axes_3d = []
    axes_intersect = {'dens': [], 'pres': [], 'vel': []}
    scatters = []

    # For each directory, add a top 3D subplot and three intersection (2D) subplots.
    for i, dataframes in enumerate(list_of_dataframes):
        # Top row: 3D scatter plot
        ax3d = fig.add_subplot(4, n_dirs, i + 1, projection="3d")
        axes_3d.append(ax3d)
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
        pos_z_all = np.concatenate([df["pos_z"].values for df in dataframes])
        ax3d.set_xlim(pos_x_all.min() - 0.05 * (pos_x_all.max() - pos_x_all.min()),
                      pos_x_all.max() + 0.05 * (pos_x_all.max() - pos_x_all.min()))
        ax3d.set_ylim(pos_y_all.min() - 0.05 * (pos_y_all.max() - pos_y_all.min()),
                      pos_y_all.max() + 0.05 * (pos_y_all.max() - pos_y_all.min()))
        ax3d.set_zlim(pos_z_all.min() - 0.05 * (pos_z_all.max() - pos_z_all.min()),
                      pos_z_all.max() + 0.05 * (pos_z_all.max() - pos_z_all.min()))
        ax3d.set_xlabel("pos_x [m]")
        ax3d.set_ylabel("pos_y [m]")
        ax3d.set_zlabel("pos_z [m]")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax3d.scatter([], [], [], s=3, c=[], cmap="viridis",
                            vmin=physics_all.min(), vmax=physics_all.max())
        scatters.append(scat)
        mappable = plt.cm.ScalarMappable(cmap="viridis")
        mappable.set_array(physics_all)
        cbar = fig.colorbar(mappable, ax=ax3d, pad=0.1)
        cbar.set_label(f"{physics_key} value")
        title = plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}"
        ax3d.set_title(f"{title} (3D)")

        # Bottom three rows: separate intersection plots.
        # Row 1 (index 1): Density, Row 2 (index 2): Pressure, Row 3 (index 3): Velocity.
        for var, row in zip(["dens", "pres", "vel"], [1, 2, 3]):
            ax = fig.add_subplot(4, n_dirs, n_dirs * row + i + 1)
            ax.set_xlabel("pos_x [m]")
            if var == "dens":
                ax.set_ylabel("Density")
            elif var == "pres":
                ax.set_ylabel("Pressure")
            elif var == "vel":
                ax.set_ylabel("Velocity")
            line, = ax.plot([], [], marker='o', linestyle='-')
            axes_intersect[var].append(ax)
            # For clarity, add a title for the intersection subplot
            ax.set_title(f"{title} {var.capitalize()} Intersection")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    n_frames = len(list_of_dataframes[0])

    def init():
        for scat in scatters:
            scat._offsets3d = (np.array([]), np.array([]), np.array([]))
        # Initialize empty data for all intersection lines.
        lines = []
        for var in axes_intersect:
            for ax in axes_intersect[var]:
                for line in ax.lines:
                    line.set_data([], [])
                    lines.append(line)
        return scatters + lines

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            # Update 3D scatter
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scatters[i]._offsets3d = (x, y, z)
            scatters[i].set_array(df[physics_key].values)
            title = plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}"
            axes_3d[i].set_title(f"{title} (3D)")

            # Intersection: select points with pos_y near the median.
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol = 0.1 * (pos_y.max() - pos_y.min())
            mask = np.abs(pos_y - median_y) < tol
            x_slice = df["pos_x"].values[mask]
            if len(x_slice) > 0:
                sorted_indices = np.argsort(x_slice)
                x_sorted = x_slice[sorted_indices]
                # Density
                dens_slice = df["dens"].values[mask][sorted_indices]
                # Pressure
                pres_slice = df["pres"].values[mask][sorted_indices]
                # Velocity (3D: compute magnitude from vel_x, vel_y, and optionally vel_z)
                if "vel_z" in df.columns:
                    vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 +
                                        df["vel_y"].values[mask] ** 2 +
                                        df["vel_z"].values[mask] ** 2)[sorted_indices]
                else:
                    vel_slice = np.sqrt(df["vel_x"].values[mask] ** 2 +
                                        df["vel_y"].values[mask] ** 2)[sorted_indices]
                # Update intersection lines for each variable.
                # Find the corresponding axes by computing subplot index: row 1 -> density, row 2 -> pressure, row 3 -> velocity.
                # Since we added one line per subplot in each bottom row, update that line.
                for var, data in zip(["dens", "pres", "vel"], [dens_slice, pres_slice, vel_slice]):
                    # Find the line in the subplot for directory i.
                    ax = fig.axes[n_dirs * ({"dens": 1, "pres": 2, "vel": 3}[var]) + i]
                    if ax.lines:
                        ax.lines[0].set_data(x_sorted, data)
                        ax.set_xlim(x_sorted.min(), x_sorted.max())
                        ax.set_ylim(data.min(), data.max())
            else:
                for var in ["dens", "pres", "vel"]:
                    ax = fig.axes[n_dirs * ({"dens": 1, "pres": 2, "vel": 3}[var]) + i]
                    if ax.lines:
                        ax.lines[0].set_data([], [])
        lines = []
        for var in axes_intersect:
            for ax in axes_intersect[var]:
                for line in ax.lines:
                    lines.append(line)
        return scatters + lines

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.show()
    return ani


def plot_intersection_along_line(df, line_start, line_end, tol=0.1, physics_keys=["dens", "vel", "pres"],
                                 show_plot=True):
    """
    Utility to plot the intersection along any arbitrary line in 2D.

    This function filters points in the provided DataFrame `df` that lie within a distance
    `tol` of the line defined by `line_start` and `line_end`. It then projects these points
    onto the line, sorts them by the distance along the line, and plots the specified physics
    variables versus the distance along the line.

    Parameters:
      - df: pandas DataFrame containing simulation data with at least "pos_x" and "pos_y".
      - line_start: tuple (x1, y1) specifying the start point of the line.
      - line_end: tuple (x2, y2) specifying the end point of the line.
      - tol: Tolerance distance for selecting points near the line.
      - physics_keys: List of physics variable names to plot. If "vel" is included, the function
                      computes the velocity magnitude from "vel_x" and "vel_y" (and "vel_z" if available).
      - show_plot: If True, displays the plot immediately.

    Returns:
      fig, ax: The matplotlib figure and axis objects.
    """
    # Extract positions
    x = df["pos_x"].values
    y = df["pos_y"].values
    points = np.column_stack((x, y))

    # Define the line vector and its norm
    p1 = np.array(line_start)
    p2 = np.array(line_end)
    d = p2 - p1
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("line_start and line_end must be different points.")

    # Compute projection parameter t for each point (scalar along the line)
    t = np.dot(points - p1, d) / (d_norm ** 2)
    # Projected points on the line
    proj = p1 + np.outer(t, d)
    # Distance from each point to its projection on the line
    dist = np.linalg.norm(points - proj, axis=1)

    # Select points within the tolerance
    mask = dist < tol
    if not np.any(mask):
        raise ValueError("No points found within the specified tolerance from the line.")

    # Compute the distance along the line (s-coordinate) for the selected points
    s = t[mask] * d_norm
    sorted_indices = np.argsort(s)
    s_sorted = s[sorted_indices]

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Loop over the physics keys and plot the corresponding data
    for key in physics_keys:
        if key == "vel":
            # Compute velocity magnitude from available velocity components
            if "vel_z" in df.columns:
                vel = np.sqrt(df["vel_x"].values[mask] ** 2 +
                              df["vel_y"].values[mask] ** 2 +
                              df["vel_z"].values[mask] ** 2)
            else:
                vel = np.sqrt(df["vel_x"].values[mask] ** 2 +
                              df["vel_y"].values[mask] ** 2)
            data = vel[sorted_indices]
            label = "Velocity magnitude"
        else:
            if key not in df.columns:
                raise ValueError(f"Column '{key}' not found in the DataFrame.")
            data = df[key].values[mask][sorted_indices]
            label = key.capitalize()
        ax.plot(s_sorted, data, marker='o', linestyle='-', label=label)

    ax.set_xlabel("Distance along line [m]")
    ax.set_ylabel("Value")
    ax.set_title("Intersection Plot along Specified Line")
    ax.legend(loc="best")
    if show_plot:
        plt.show()
    return fig, ax


# --- Main function ---
def main():
    # Choose the plot type: "1d", "2d", or "3d"
    # For 2D and 3D, the figures now include the original plot and separate intersection rows.
    plot_type = "3d"  # Change to "1d", "2d", or "3d" as needed

    # List the data directories for comparison
    data_dirs = [
        "/Users/guo/research/sim_result_vis/result_data/khi_disph/results",
        "/Users/guo/research/sim_result_vis/result_data/khi_ssph/results",
    ]
    # Supply custom titles for each plot (optional)
    plot_titles = ["Density Independent SPH", "Standard SPH"]
    physics_var = "dens"  # default physics key for the original colored plot

    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        if plot_type in ["1d", "2d"]:
            dfs, times = load_dataframes(data_path)
        elif plot_type == "3d":
            dfs, times = load_dataframes_3d(data_path)
        else:
            raise ValueError("Invalid plot_type. Choose '1d', '2d', or '3d'.")
        list_of_dataframes.append(dfs)
        list_of_times.append(times)

    # For simplicity we assume that all directories have the same number of frames.
    if plot_type == "1d":
        animate_multiple_1d(list_of_dataframes, list_of_times, plot_titles=plot_titles)
    elif plot_type == "2d":
        animate_multiple_2d(list_of_dataframes, list_of_times, physics_key=physics_var, plot_titles=plot_titles)
    elif plot_type == "3d":
        animate_multiple_3d(list_of_dataframes, list_of_times, physics_key=physics_var, plot_titles=plot_titles)

    # Example usage of the intersection utility on the first frame of the first dataset:
    try:
        df_example = list_of_dataframes[0][0]
        # Define a line by its start and end points (modify as needed)
        line_start = (df_example["pos_x"].min(), df_example["pos_y"].min())
        line_end = (df_example["pos_x"].max(), df_example["pos_y"].max())
        plot_intersection_along_line(df_example, line_start, line_end, tol=0.1,
                                     physics_keys=["dens", "vel", "pres"])
    except Exception as e:
        print("Error in plotting intersection along line:", e)


if __name__ == "__main__":
    main()
