# common.py
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_THRESHOLD = 1e5  # adjust this threshold as needed
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


def load_dataframes_1d(data_path, file_extension="*.dat"):
    """
    Load 1D data files from the specified directory.

    Expected 1D files have 12 columns:
      ["pos_x", "vel_x", "acc_x", "mass", "dens", "pres", "ene", "sml", "id", "neighbor", "alpha", "gradh"]

    Returns:
      (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 1D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 1D files in {data_path}.")

    times = [extract_time(file) for file in file_list]
    columns = [
        "pos_x", "vel_x", "acc_x",
        "mass", "dens", "pres", "ene",
        "sml", "id", "neighbor", "alpha", "gradh"
    ]
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, names=columns)
        # Debug: Check for unusually large pos_x values.
        large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
        if large_indices:
            print(f"DEBUG: In file '{file}', large pos_x at rows {large_indices}")
        dataframes.append(df)
    return dataframes, times


def load_dataframes_2d(data_path, file_extension="*.dat"):
    """
    Load 2D data files from the specified directory.

    Expected 2D files have 15 columns:
      ["pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y",
       "mass", "dens", "pres", "ene", "sml", "id", "neighbor", "alpha", "gradh"]

    Returns:
      (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 2D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 2D files in {data_path}.")

    times = [extract_time(file) for file in file_list]
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
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, names=columns)
        large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
        if large_indices:
            print(f"DEBUG: In file '{file}', large pos_x at rows {large_indices}")
        dataframes.append(df)
    return dataframes, times


def load_dataframes_3d(data_path, file_extension="*.dat"):
    """
    Load 3D data files from the specified directory.

    Supports files with:
      - 15 columns (often missing acceleration components),
      - 18 columns (standard 3D), or
      - 33 columns (with additional gradient info).

    Returns:
      (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 3D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 3D files in {data_path}.")

    times = [extract_time(file) for file in file_list]
    dataframes = []
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
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None)
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
            df.rename(columns=dict(zip(range(ncols), basic_columns[:ncols])), inplace=True)
            print(f"Warning: File '{file}' has {ncols} columns; expected 15, 18, or 33 columns.")
        large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
        if large_indices:
            print(f"DEBUG: In file '{file}', large pos_x at rows {large_indices}")
        dataframes.append(df)
    return dataframes, times


def add_sph_formulas(ax):
    """
    Add SPH formulas text to an axis.
    """
    text = (
            r"Standard SPH: $\rho_i=\sum_j m_j\,W(|\mathbf{r}_i-\mathbf{r}_j|, h)$" "\n" +
            r"Density Ind. SPH: $P_i=P_0(\hat{\rho}_i/\rho_0)^\gamma$, "
            r"$\hat{\rho}_i=\sum_j m_j (A_j/A_i)^{1/\gamma}\,W(|\mathbf{r}_i-\mathbf{r}_j|, h)$"
    )
    ax.text(0.05, -0.55, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7))


def plot_intersection_along_line(df, line_start, line_end, tol=0.1, physics_keys=["dens", "vel", "pres"],
                                 show_plot=True):
    """
    Utility to plot the intersection along an arbitrary line in 2D.

    Parameters:
      - df: DataFrame with at least "pos_x" and "pos_y"
      - line_start, line_end: tuples (x, y) defining the line
      - tol: tolerance distance from the line
      - physics_keys: list of variables to plot; if "vel" is present, computes velocity magnitude.
      - show_plot: if True, displays the plot.

    Returns:
      fig, ax: the figure and axis objects.
    """
    x = df["pos_x"].values
    y = df["pos_y"].values
    points = np.column_stack((x, y))
    p1 = np.array(line_start)
    p2 = np.array(line_end)
    d = p2 - p1
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("line_start and line_end must be different.")
    # Compute projection parameter t for each point (scalar along the line)
    t = np.dot(points - p1, d) / (d_norm ** 2)
    proj = p1 + np.outer(t, d)
    dist = np.linalg.norm(points - proj, axis=1)
    mask = dist < tol
    if not np.any(mask):
        raise ValueError("No points found within tolerance.")
    s = t[mask] * d_norm
    sorted_indices = np.argsort(s)
    s_sorted = s[sorted_indices]
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in physics_keys:
        if key == "vel":
            if "vel_z" in df.columns:
                vel = np.sqrt(
                    df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2 + df["vel_z"].values[mask] ** 2)
            else:
                vel = np.sqrt(df["vel_x"].values[mask] ** 2 + df["vel_y"].values[mask] ** 2)
            data = vel[sorted_indices]
            label = "Velocity magnitude"
        else:
            if key not in df.columns:
                raise ValueError(f"Column '{key}' not in DataFrame.")
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
