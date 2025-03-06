import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from shocktube.sod_shocktube_1d.shocktube_strong_analytic_1d import analytic_sod_strong_solution_1d
from vis.common import load_dataframes_1d, generate_title_from_dir, load_units_json


def animate_multiple_1d(list_of_dataframes, list_of_times, plot_titles=None, units_data=None):
    """
    Animate 1D scatter plots with analytic solution overlay.
    Axis labels are automatically updated with units extracted from the CSV header.

    The analytic solver is passed the same units_data so that its output is scaled.
    """
    n_dirs = len(list_of_dataframes)
    fig, axes = plt.subplots(3, n_dirs, figsize=(5 * n_dirs, 12), sharex=True)
    if n_dirs == 1:
        axes = axes.reshape(3, 1)

    scatters = {'dens': [], 'pres': [], 'vel_x': []}
    lines = {'dens': [], 'pres': [], 'vel_x': []}
    gamma = 1.6666666  # Use the same gamma as in your simulation configuration

    for i, dataframes in enumerate(list_of_dataframes):
        # Get unit mapping from the first dataframe of the dataset.
        units = dataframes[0].attrs.get("units", {}) if dataframes else {}
        dens_unit = units.get("dens", "kg/m³")
        pres_unit = units.get("pres", "Pa")
        vel_unit = units.get("vel_x", "m/s")
        pos_unit = units.get("pos_x", "m")

        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        dens_all = np.concatenate([df["dens"].values for df in dataframes])
        press_all = np.concatenate([df["pres"].values for df in dataframes])
        vel_x_all = np.concatenate([df["vel_x"].values for df in dataframes])

        # Density subplot
        ax_dens = axes[0, i]
        ax_dens.set_xlim(pos_x_all.min() * 0.95, pos_x_all.max() * 1.05)
        ax_dens.set_ylim(dens_all.min() * 0.95, dens_all.max() * 1.05)
        ax_dens.set_ylabel(f"dens [{dens_unit}]")
        scat_dens = ax_dens.scatter([], [], s=10, c="blue", label='Simulation')
        line_dens, = ax_dens.plot([], [], 'k-', label='Analytic')
        scat_dens.set_animated(True)
        line_dens.set_animated(True)
        scatters['dens'].append(scat_dens)
        lines['dens'].append(line_dens)
        ax_dens.set_title(plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")
        ax_dens.legend()

        # Pressure subplot
        ax_press = axes[1, i]
        ax_press.set_ylim(press_all.min() * 0.95, press_all.max() * 1.05)
        ax_press.set_ylabel(f"pres [{pres_unit}]")
        scat_press = ax_press.scatter([], [], s=10, c="red", label='Simulation')
        line_press, = ax_press.plot([], [], 'k-', label='Analytic')
        scat_press.set_animated(True)
        line_press.set_animated(True)
        scatters['pres'].append(scat_press)
        lines['pres'].append(line_press)
        ax_press.legend()

        # Velocity subplot
        ax_vel = axes[2, i]
        ax_vel.set_ylim(vel_x_all.min() * 0.95, vel_x_all.max() * 1.05)
        ax_vel.set_ylabel(f"vel_x [{vel_unit}]")
        ax_vel.set_xlabel(f"pos_x [{pos_unit}]")
        scat_vel = ax_vel.scatter([], [], s=10, c="green", label='Simulation')
        line_vel, = ax_vel.plot([], [], 'k-', label='Analytic')
        scat_vel.set_animated(True)
        line_vel.set_animated(True)
        scatters['vel_x'].append(scat_vel)
        lines['vel_x'].append(line_vel)
        ax_vel.legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        for var in scatters:
            for scat in scatters[var]:
                scat.set_offsets(np.empty((0, 2)))
            for line in lines[var]:
                line.set_data([], [])
        return [item for var in scatters for item in scatters[var] + lines[var]]

    def update(frame_index):
        all_items = []
        # Loop over each dataset/direction
        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            sim_x = df["pos_x"].values
            t = times[frame_index]

            # Update simulation scatter points.
            scatters['dens'][i].set_offsets(np.column_stack((sim_x, df["dens"].values)))
            scatters['pres'][i].set_offsets(np.column_stack((sim_x, df["pres"].values)))
            scatters['vel_x'][i].set_offsets(np.column_stack((sim_x, df["vel_x"].values)))

            # Create a uniform grid for the analytic solution.
            x_grid = np.linspace(sim_x.min(), sim_x.max(), 500)
            # Compute analytic solution with unit scaling.
            rho_analytic, p_analytic, v_analytic = analytic_sod_strong_solution_1d(
                x_grid, t, gamma)
            lines['dens'][i].set_data(x_grid, rho_analytic)
            lines['pres'][i].set_data(x_grid, p_analytic)
            lines['vel_x'][i].set_data(x_grid, v_analytic)

            all_items.extend([scatters['dens'][i], lines['dens'][i],
                              scatters['pres'][i], lines['pres'][i],
                              scatters['vel_x'][i], lines['vel_x'][i]])
        return all_items

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


def main():


    data_dirs = [
        "/Users/guo/OSS/sphcode/sample/shock_tube_strong_shock/results/SSPH/shock_tube_strong_shock/1D",
        "/Users/guo/OSS/sphcode/sample/shock_tube_strong_shock/results/GSPH/shock_tube_strong_shock/1D",
        "/Users/guo/OSS/sphcode/sample/shock_tube_strong_shock/results/DISPH/shock_tube_strong_shock/1D",
        "/Users/guo/OSS/sphcode/sample/shock_tube_strong_shock/results/GDISPH/shock_tube_strong_shock/1D",

    ]
    # Specify the path to your units JSON file.
    units_json_path = "/Users/guo/OSS/sphcode/sample/shock_tube_strong_shock/units.json"
    # Load the units data.
    units_data = load_units_json(units_json_path)
    plot_titles = [generate_title_from_dir(d) for d in data_dirs]
    list_of_dataframes = []
    list_of_times = []
    # Load simulation data with unit scaling.
    for data_path in data_dirs:
        dfs, times = load_dataframes_1d(data_path, units_json_path=units_json_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    # Start the animation; analytic solver receives the same units_data.
    animate_multiple_1d(list_of_dataframes, list_of_times, plot_titles=plot_titles, units_data=units_data)


if __name__ == "__main__":
    main()
