# old plot utils
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_dataframes(data_path, file_extension="*.dat"):
    """
    Load all .dat files from the specified directory into a list of DataFrames.

    Each file is read with the same column names based on the header.
    """
    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    if not file_list:
        raise FileNotFoundError(f"No files found in directory: {data_path}")
    print(f"Found {len(file_list)} files.")

    # Define column names based on the file header
    columns = ["pos", "vel", "acc", "mass", "dens", "pres", "ene", "sml", "id", "neighbor", "alpha", "gradh"]

    # Read each file into a DataFrame (skip header lines starting with '#')
    dataframes = [
        pd.read_csv(file, delim_whitespace=True, comment='#', header=None, names=columns)
        for file in file_list
    ]
    return dataframes


def compute_global_limits(dataframes, x_key="pos", y_key="dens", padding=0.05):
    """
    Compute the global minimum and maximum for x and y values across all DataFrames,
    adding a small percentage as padding.
    """
    x_min = min(df[x_key].min() for df in dataframes)
    x_max = max(df[x_key].max() for df in dataframes)
    y_min = min(df[y_key].min() for df in dataframes)
    y_max = max(df[y_key].max() for df in dataframes)

    # Calculate ranges and add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    return x_min, x_max, y_min, y_max


def animate_data(dataframes):
    """
    Create a 1D animation of the data using a scatter plot.

    The x-axis corresponds to the 'pos' column and the y-axis to the 'dens' column.
    """
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=50)  # Adjust marker size as needed

    # Compute global axis limits and set them
    x_min, x_max, y_min, y_max = compute_global_limits(dataframes)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    def init():
        # Initialize with an empty scatter plot
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame_index):
        # Update the scatter plot with data from the current frame
        df = dataframes[frame_index]
        x = df["pos"].values
        y = df["dens"].values
        offsets = np.column_stack((x, y))
        scat.set_offsets(offsets)
        ax.set_title(f"Frame {frame_index + 1} of {len(dataframes)}")
        return scat,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(dataframes), init_func=init, interval=100, blit=True)
    plt.show()
    return ani


def main():
    # Specify the directory containing your .dat files (update this path as needed)
    data_path = "/Users/guo/OSS/sphcode/results"

    # Load data and create the animation
    dataframes = load_dataframes(data_path)
    animate_data(dataframes)


if __name__ == "__main__":
    main()
