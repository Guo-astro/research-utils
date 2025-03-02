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

    Two possible formats are supported:
      - Basic format: 15 columns with the following header:
           pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, mass, dens, pres, ene, sml, id, neighbor, alpha, gradh
      - Extended format: 23 columns (15 basic + 8 additional columns) where the additional columns correspond to:
           grad_velocity_0 (vector with 2 components),
           grad_pressure (vector with 2 components),
           grad_velocity_1 (vector with 2 components),
           grad_density (vector with 2 components)

    Returns: (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 2D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 2D files.")

    times = [extract_time(file) for file in file_list]

    dataframes = []
    for file in file_list:
        # Read file without assigning column names yet.
        df = pd.read_csv(file, sep='\s+', comment='#', header=None)
        ncols = df.shape[1]

        if ncols == 15:
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
        elif ncols == 23:
            basic_columns = [
                "pos_x", "pos_y",
                "vel_x", "vel_y",
                "acc_x", "acc_y",
                "mass", "dens",
                "pres", "ene",
                "sml", "id",
                "neighbor", "alpha",
                "gradh"
            ]
            additional_columns = [
                "grad_velocity_0_x", "grad_velocity_0_y",
                "grad_pressure_x", "grad_pressure_y",
                "grad_velocity_1_x", "grad_velocity_1_y",
                "grad_density_x", "grad_density_y"
            ]
            columns = basic_columns + additional_columns
        else:
            print(
                f"Warning: File '{file}' has {ncols} columns; expected 15 or 23 columns. Using first 15 columns with basic names.")
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
            # Optionally, you could slice the dataframe: df = df.iloc[:, :15]

        df.columns = columns

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

    The files are assumed to be whitespace‐separated and to contain comment lines.

    There are two supported formats:
      1. Basic format with 18 columns (expected header):
         pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
         mass, dens, pres, ene, sml, id, neighbor, alpha, gradh
      2. Extended format with additional arrays, having a total of 33 columns:
         The first 18 columns are as above. The additional 15 columns correspond to:
           grad_velocity_0 (vector: x,y,z),
           grad_pressure (vector: x,y,z),
           grad_velocity_2 (vector: x,y,z),
           grad_velocity_1 (vector: x,y,z),
           grad_density (vector: x,y,z)
    Returns: (dataframes, times)
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 3D files found in directory: {data_path}")
    print(f"Found {len(file_list)} 3D files.")

    times = [extract_time(file) for file in file_list]

    dataframes = []
    for file in file_list:
        # Read all columns (do not specify names so that we can check number of columns)
        df = pd.read_csv(file, sep='\s+', comment='#', header=None)
        ncols = df.shape[1]

        # Define the basic columns (first 18)
        basic_columns = [
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z",
            "mass", "dens", "pres", "ene",
            "sml", "id", "neighbor", "alpha", "gradh"
        ]

        if ncols == 18:
            df.columns = basic_columns
        elif ncols == 33:
            additional_columns = [
                "grad_velocity_0_x", "grad_velocity_0_y", "grad_velocity_0_z",
                "grad_pressure_x", "grad_pressure_y", "grad_pressure_z",
                "grad_velocity_2_x", "grad_velocity_2_y", "grad_velocity_2_z",
                "grad_velocity_1_x", "grad_velocity_1_y", "grad_velocity_1_z",
                "grad_density_x", "grad_density_y", "grad_density_z"
            ]
            df.columns = basic_columns + additional_columns
        else:
            df.rename(columns=dict(zip(range(18), basic_columns)), inplace=True)
            print(f"Warning: File '{file}' has {ncols} columns; expected 18 or 33 columns.")

        # Debug: Check for unusually large pos_x values.
        large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
        if large_indices:
            print(f"DEBUG: In file '{file}', found large pos_x values at rows {large_indices}:")
            for idx in large_indices:
                print(f"  Row {idx}: pos_x = {df.loc[idx, 'pos_x']}")
        dataframes.append(df)
    return dataframes, times


# Animation functions remain the same as before.
def animate_1d(dataframes, times):
    fig, ax = plt.subplots()
    pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
    dens_all = np.concatenate([df["dens"].values for df in dataframes])
    ax.set_xlim(pos_x_all.min() - 0.05 * (pos_x_all.max() - pos_x_all.min()),
                pos_x_all.max() + 0.05 * (pos_x_all.max() - pos_x_all.min()))
    ax.set_ylim(dens_all.min() - 0.05 * (dens_all.max() - dens_all.min()),
                dens_all.max() + 0.05 * (dens_all.max() - dens_all.min()))
    ax.set_xlabel("pos_x [m]")
    ax.set_ylabel("dens [kg/m³]")
    scat = ax.scatter([], [], s=10, c="blue")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame_index):
        df = dataframes[frame_index]
        x = df["pos_x"].values
        y = df["dens"].values
        scat.set_offsets(np.column_stack((x, y)))
        time_str = f"{times[frame_index]:.3f} s" if times[frame_index] is not None else "N/A"
        ax.set_title(f"1D Plot: Frame {frame_index + 1}/{len(dataframes)} | Time: {time_str}")
        return scat,

    ani = FuncAnimation(fig, update, frames=len(dataframes), init_func=init, interval=100, blit=True)
    plt.show()
    return ani


def animate_2d(dataframes, times, physics_key="dens"):
    fig, ax = plt.subplots()
    pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
    pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
    ax.set_xlim(pos_x_all.min() - 0.05 * (pos_x_all.max() - pos_x_all.min()),
                pos_x_all.max() + 0.05 * (pos_x_all.max() - pos_x_all.min()))
    ax.set_ylim(pos_y_all.min() - 0.05 * (pos_y_all.max() - pos_y_all.min()),
                pos_y_all.max() + 0.05 * (pos_y_all.max() - pos_y_all.min()))
    ax.set_xlabel("pos_x [m]")
    ax.set_ylabel("pos_y [m]")

    physics_all = np.concatenate([df[physics_key].values for df in dataframes])
    scat = ax.scatter([], [], s=30, c=[], cmap="viridis", vmin=physics_all.min(), vmax=physics_all.max())
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(f"{physics_key} value")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame_index):
        df = dataframes[frame_index]
        x = df["pos_x"].values
        y = df["pos_y"].values
        colors = df[physics_key].values
        scat.set_offsets(np.column_stack((x, y)))
        scat.set_array(colors)
        time_str = f"{times[frame_index]:.3f} s" if times[frame_index] is not None else "N/A"
        ax.set_title(f"2D Plot: Frame {frame_index + 1}/{len(dataframes)} | Time: {time_str} (color: {physics_key})")
        return scat,

    ani = FuncAnimation(fig, update, frames=len(dataframes), init_func=init, interval=100, blit=True)
    plt.show()
    return ani


def animate_3d(dataframes, times, physics_key="dens"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
    pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
    pos_z_all = np.concatenate([df["pos_z"].values for df in dataframes])
    ax.set_ylim(pos_y_all.min() - 0.05 * (pos_y_all.max() - pos_y_all.min()),
                pos_y_all.max() + 0.05 * (pos_y_all.max() - pos_y_all.min()))
    ax.set_zlim(pos_z_all.min() - 0.05 * (pos_z_all.max() - pos_z_all.min()),
                pos_z_all.max() + 0.05 * (pos_z_all.max() - pos_z_all.min()))
    ax.set_xlabel("pos_x [m]")
    ax.set_ylabel("pos_y [m]")
    ax.set_zlabel("pos_z [m]")

    physics_all = np.concatenate([df[physics_key].values for df in dataframes])
    scat = ax.scatter([], [], [], s=3, c=[], cmap="viridis")
    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(physics_all)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label(f"{physics_key} value")

    def init():
        scat._offsets3d = (np.array([]), np.array([]), np.array([]))
        return scat,

    def update(frame_index):
        df = dataframes[frame_index]
        x = df["pos_x"].values
        y = df["pos_y"].values
        z = df["pos_z"].values
        scat._offsets3d = (x, y, z)
        scat.set_array(df[physics_key].values)
        time_str = f"{times[frame_index]:.3f} s" if times[frame_index] is not None else "N/A"
        ax.set_title(f"3D Plot: Frame {frame_index + 1}/{len(dataframes)} | Time: {time_str} (color: {physics_key})")
        return scat,

    ani = FuncAnimation(fig, update, frames=len(dataframes), init_func=init, interval=100, blit=False)
    plt.show()
    return ani


def main():
    # Select the plot type: "1d", "2d", or "3d"
    plot_type = "3d"  # Change this to "1d", "2d" or "3d" as needed

    # Set the data directory and the physics variable for coloring (if applicable)
    data_path = "/Users/guo/OSS/sphcode/results"
    physics_var = "dens"  # e.g. "vel_x", "pres", etc.

    if plot_type == "1d":
        dataframes, times = load_dataframes(data_path)
        animate_1d(dataframes, times)
    elif plot_type == "2d":
        dataframes, times = load_dataframes(data_path)
        animate_2d(dataframes, times, physics_key=physics_var)
    elif plot_type == "3d":
        dataframes, times = load_dataframes_3d(data_path)
        animate_3d(dataframes, times, physics_key=physics_var)
    else:
        print("Invalid plot_type selected. Choose '1d', '2d', or '3d'.")


if __name__ == "__main__":
    main()
