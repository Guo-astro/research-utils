import os
import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_THRESHOLD = 1e5  # adjust this threshold as needed
EXCLUDE_PATTERNS = ["energy", ".log"]


def load_units_json(filename="units.json"):
    """
    Load the units and conversion factors from a JSON file.
    Expected keys include:
      - time_factor, length_factor, mass_factor, density_factor,
        pressure_factor, energy_factor,
      - time_unit, length_unit, mass_unit, density_unit,
        pressure_unit, energy_unit,
      - velocity_unit: desired velocity unit for plotting,
      - plot_velocity_conversion: factor to convert simulation velocity (m/s)
        to desired plotting unit (e.g. km/s).
    """
    with open(filename, "r") as f:
        return json.load(f)


# Load the default unit settings from units.json.

def parse_column_units(columns):
    """
    Given a list of column names (strings) like "time [s]", "pos_x [m]", etc.,
    return a tuple (new_columns, units_mapping) where:
      - new_columns is a list of the base names (e.g. "time", "pos_x")
      - units_mapping is a dict mapping the base name to its unit (e.g. {"time": "s", "pos_x": "m"})
    """
    new_columns = []
    units_mapping = {}
    for col in columns:
        m = re.match(r"(.+?)\s*\[(.+?)\]", col)
        if m:
            base = m.group(1).strip()
            unit = m.group(2).strip()
            new_columns.append(base)
            units_mapping[base] = unit
        else:
            new_columns.append(col)
            units_mapping[col] = ""
    return new_columns, units_mapping


def extract_time(file):
    """
    Extract the simulation time from the CSV file.
    Assumes that the CSV file has a header row with a "time" column.
    """
    try:
        # Read only the first row to get the time
        df = pd.read_csv(file, nrows=1)
        new_cols, units = parse_column_units(df.columns)
        df.columns = new_cols
        df.attrs["units"] = units
        if "time" in df.columns:
            return float(df["time"].iloc[0])
    except Exception as e:
        print(f"Error reading time from file {file}: {e}")
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


def apply_scaling_to_dataframe(df, units_data):
    """
    Apply unit conversions to the DataFrame using the provided units_data.
    For example, convert the velocity columns from m/s (simulation output)
    to the desired unit (e.g. km/s) using the "plot_velocity_conversion" factor.
    Also update the units in df.attrs["units"] accordingly.
    """
    units = df.attrs.get("units", {})
    if "vel_x" in df.columns and "plot_velocity_conversion" in units_data:
        # Convert velocity from m/s to desired plotting unit.
        conversion = units_data["plot_velocity_conversion"]
        df["vel_x"] = df["vel_x"] * conversion
        # Update unit label.
        units["vel_x"] = units_data.get("velocity_unit", "km/s")
    df.attrs["units"] = units


def load_dataframes_1d(data_path, file_extension="*.csv", units_json_path=None):
    """
    Load 1D CSV data files from the specified directory.
    Expects a header line such as:
      time [s], pos_x [m], vel_x [m/s], acc_x [m/s^2], mass [kg], dens [kg/m^3],
      pres [Pa], ene [J/kg], sml [m], id, neighbor, alpha, gradh, ...
    Returns:
      (dataframes, times)
    """

    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 1D CSV files found in directory: {data_path}")
    print(f"Found {len(file_list)} 1D CSV files in {data_path}.")

    dataframes = []
    times = []
    for file in file_list:
        df = pd.read_csv(file)  # header is read automatically
        new_cols, units = parse_column_units(df.columns)
        df.columns = new_cols
        df.attrs["units"] = units

        # Apply scaling conversions (e.g. velocity)
        # apply_scaling_to_dataframe(df, units_data)

        if "time" in df.columns:
            times.append(df["time"].iloc[0])
        else:
            times.append(None)
        if "pos_x" in df.columns:
            large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
            if large_indices:
                print(f"DEBUG: In file '{file}', large pos_x at rows {large_indices}")
        dataframes.append(df)
    return dataframes, times


def load_dataframes_2d(data_path, file_extension="*.csv", units_json_path=None):
    """
    Load 2D CSV data files from the specified directory.
    Expects a header line such as:
      time [s], pos_x [m], pos_y [m], vel_x [m/s], vel_y [m/s], acc_x [m/s^2],
      acc_y [m/s^2], mass [kg], dens [kg/m^3], pres [Pa], ene [J/kg],
      sml [m], id, neighbor, alpha, gradh, ...
    Returns:
      (dataframes, times)
    """

    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 2D CSV files found in directory: {data_path}")
    print(f"Found {len(file_list)} 2D CSV files in {data_path}.")

    dataframes = []
    times = []
    for file in file_list:
        df = pd.read_csv(file)
        new_cols, units = parse_column_units(df.columns)
        df.columns = new_cols
        df.attrs["units"] = units

        # apply_scaling_to_dataframe(df, units_data)

        if "time" in df.columns:
            times.append(df["time"].iloc[0])
        else:
            times.append(None)
        if "pos_x" in df.columns:
            large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
            if large_indices:
                print(f"DEBUG: In file '{file}', large pos_x at rows {large_indices}")
        dataframes.append(df)
    return dataframes, times


def load_dataframes_3d(data_path, file_extension="*.csv", units_json_path=None):
    """
    Load 3D CSV data files from the specified directory.
    Expects a header line such as:
      time [s], pos_x [m], pos_y [m], pos_z [m], vel_x [m/s], vel_y [m/s],
      vel_z [m/s], acc_x [m/s^2], acc_y [m/s^2], acc_z [m/s^2], mass [kg],
      dens [kg/m^3], pres [Pa], ene [J/kg], sml [m], id, neighbor, alpha, gradh, ...
    Returns:
      (dataframes, times)
    """

    file_list = sorted(glob.glob(os.path.join(data_path, file_extension)))
    file_list = filter_files(file_list)
    if not file_list:
        raise FileNotFoundError(f"No 3D CSV files found in directory: {data_path}")
    print(f"Found {len(file_list)} 3D CSV files in {data_path}.")

    dataframes = []
    times = []
    for file in file_list:
        df = pd.read_csv(file)
        new_cols, units = parse_column_units(df.columns)
        df.columns = new_cols
        df.attrs["units"] = units


        if "time" in df.columns:
            times.append(df["time"].iloc[0])
        else:
            times.append(None)
        if "pos_x" in df.columns:
            large_indices = df.index[np.abs(df["pos_x"]) > DEBUG_THRESHOLD].tolist()
            if large_indices:
                print(f"DEBUG: In file '{file}', large pos_x at rows {large_indices}")
        dataframes.append(df)
    return dataframes, times


def plot_intersection_along_line(df, line_start, line_end, tol=0.1,
                                 physics_keys=["dens", "vel", "pres"], show_plot=True):
    """
    Utility to plot the intersection along an arbitrary line in 2D.
    The axis labels include units as read from the CSV header.

    Parameters:
      - df: DataFrame with at least "pos_x" and "pos_y"
      - line_start, line_end: tuples (x, y) defining the line
      - tol: tolerance distance from the line
      - physics_keys: list of variables to plot; if "vel" is present, computes velocity magnitude.
      - show_plot: if True, displays the plot.

    Returns:
      fig, ax: the figure and axis objects.
    """
    # Try to get the units mapping from df.attrs
    units = df.attrs.get("units", {})

    x = df["pos_x"].values
    y = df["pos_y"].values
    points = np.column_stack((x, y))
    p1 = np.array(line_start)
    p2 = np.array(line_end)
    d = p2 - p1
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("line_start and line_end must be different.")
    # Compute projection parameter t for each point.
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
            # Compute velocity magnitude using available velocity components.
            if "vel_z" in df.columns:
                vel = np.sqrt(df["vel_x"].values[mask] ** 2 +
                              df["vel_y"].values[mask] ** 2 +
                              df["vel_z"].values[mask] ** 2)
            else:
                vel = np.sqrt(df["vel_x"].values[mask] ** 2 +
                              df["vel_y"].values[mask] ** 2)
            data = vel[sorted_indices]
            # Use the updated unit from 'vel_x' for velocity.
            vel_unit = units.get("vel_x", "km/s")
            label = f"Velocity magnitude [{vel_unit}]"
        else:
            if key not in df.columns:
                raise ValueError(f"Column '{key}' not in DataFrame.")
            data = df[key].values[mask][sorted_indices]
            key_unit = units.get(key, "")
            label = f"{key.capitalize()} [{key_unit}]" if key_unit else key.capitalize()
        ax.plot(s_sorted, data, marker='o', linestyle='-', label=label)

    # Set x-axis label using the unit from pos_x (if available)
    posx_unit = units.get("pos_x", "m")
    ax.set_xlabel(f"Distance along line [{posx_unit}]")
    ax.set_ylabel("Value")
    ax.set_title("Intersection Plot along Specified Line")
    ax.legend(loc="best")
    if show_plot:
        plt.show()
    return fig, ax


def generate_title_from_dir(data_dir):
    """
    Automatically extract the SPH type from the data directory.
    Assumes the directory structure is:
      .../results/<SPH_TYPE>/<sample_name>/1D
    This function returns the folder name corresponding to <SPH_TYPE>.
    """
    parent = os.path.dirname(os.path.dirname(data_dir))
    return os.path.basename(parent)
