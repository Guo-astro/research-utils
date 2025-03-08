#!/usr/bin/env python3
import math
import csv
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from shocktube.sod_shocktube_1d.shocktube_analytic_1d import analytic_sod_solution_1d
from vis.common import load_dataframes_2d, generate_title_from_dir

# ---------------------------------------------------------------------
# Analytic Lane–Emden functions (for n = 3/2, using 3D formulation)
# ---------------------------------------------------------------------
# Global constant: first zero for n = 3/2 (3D version).
XI1_GLOBAL = 3.65375


def compute_lane_emden_n32(num_steps=10000):
    """
    Compute the Lane–Emden solution for n = 3/2 using RK4 from xi = 1e-6 to XI1_GLOBAL.
    Returns:
      xi_arr: numpy array of xi values.
      theta_arr: numpy array of theta values.
    """
    xi = 1e-6
    theta = 1.0 - (xi ** 2) / 6.0  # series expansion for 3D
    dtheta = -xi / 3.0
    h = (XI1_GLOBAL - 1e-6) / num_steps

    xi_list = [xi]
    theta_list = [theta]

    for _ in range(num_steps):
        k1_theta = h * dtheta
        k1_dtheta = h * (- math.pow(theta, 1.5) - (2.0 / xi) * dtheta)

        k2_theta = h * (dtheta + k1_dtheta / 2)
        k2_dtheta = h * (- math.pow(theta + k1_theta / 2, 1.5) -
                         (2.0 / (xi + h / 2)) * (dtheta + k1_dtheta / 2))

        k3_theta = h * (dtheta + k2_dtheta / 2)
        k3_dtheta = h * (- math.pow(theta + k2_theta / 2, 1.5) -
                         (2.0 / (xi + h / 2)) * (dtheta + k2_dtheta / 2))

        k4_theta = h * (dtheta + k3_dtheta)
        k4_dtheta = h * (- math.pow(theta + k3_theta, 1.5) -
                         (2.0 / (xi + h)) * (dtheta + k3_dtheta))

        theta += (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6.0
        dtheta += (k1_dtheta + 2 * k2_dtheta + 2 * k3_dtheta + k4_dtheta) / 6.0
        xi += h

        xi_list.append(xi)
        theta_list.append(theta)

    return np.array(xi_list), np.array(theta_list)


def lane_emden_analytic_n32(r_max, dens_c, pres_c, num_steps=10000):
    """
    Compute the analytic Lane–Emden solution for n = 3/2.

    The solution is computed in xi-space (xi from 1e-6 to XI1_GLOBAL) and then scaled:
      r = a * xi,  with a = r_max / XI1_GLOBAL.

    The analytic density and pressure are given by:
      rho(r) = dens_c * theta^(3/2)
      P(r)   = pres_c  * theta^(5/2)

    Returns:
      r_analytic: radial positions.
      analytic_dens: density profile.
      analytic_pres: pressure profile.
    """
    xi_arr, theta_arr = compute_lane_emden_n32(num_steps=num_steps)
    a = r_max / XI1_GLOBAL
    r_analytic = xi_arr * a
    analytic_dens = dens_c * np.power(theta_arr, 1.5)
    analytic_pres = pres_c * np.power(theta_arr, 2.5)
    return r_analytic, analytic_dens, analytic_pres


# ---------------------------------------------------------------------
# Existing analytic functions for x- and y-slices (from shocktube)
# ---------------------------------------------------------------------
def analytic_density_profile(x):
    """Analytic density profile for x-slice: perfect match (central region = 2)."""
    return 2.0 * np.ones_like(x)


def analytic_pressure_profile(x):
    """Analytic pressure profile for x-slice: constant 2.5."""
    return 2.5 * np.ones_like(x)


def analytic_velocity_profile(x):
    """Analytic velocity profile for x-slice: constant 0.5 (from region 2)."""
    return 0.5 * np.ones_like(x)


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


# ---------------------------------------------------------------------
# Animation function (2D version) with radial profiles
# ---------------------------------------------------------------------
def animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None):
    """
    Create an animation with multiple rows per dataset:
      Row 0: Scatter plot (pos_x vs pos_y) with a common color bar.
      Rows 1-3: x-slice plots (pos_x vs density, pressure, velocity) with analytic overlay.
      Rows 4-6: y-slice plots (pos_y vs density, pressure, velocity) with analytic overlay.
      Rows 7-9: Radial profiles (radius vs density, pressure, and velocity).
                 Analytic Lane–Emden profiles (n=3/2) are overlaid on density and pressure.
    """
    n_dirs = len(list_of_dataframes)

    # --- Precompute global limits for scatter ---
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

    # --- Precompute global limits for x-slice ---
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

    # --- Precompute global limits for y-slice ---
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

    # --- Precompute global limits for radial profiles ---
    all_r = []
    all_dens_rad = []
    all_pres_rad = []
    all_vel_rad = []
    for dataframes in list_of_dataframes:
        for df in dataframes:
            r = np.sqrt(df["pos_x"].values ** 2 + df["pos_y"].values ** 2)
            all_r.append(r)
            all_dens_rad.append(df["dens"].values)
            all_pres_rad.append(df["pres"].values)
            vel = np.sqrt(df["vel_x"].values ** 2 + df["vel_y"].values ** 2)
            all_vel_rad.append(vel)
    if all_r:
        all_r = np.concatenate(all_r)
        all_dens_rad = np.concatenate(all_dens_rad)
        all_pres_rad = np.concatenate(all_pres_rad)
        all_vel_rad = np.concatenate(all_vel_rad)
        radial_x_lim = (all_r.min() * 0.95, all_r.max() * 1.05)
        radial_dens_lim = (all_dens_rad.min(), all_dens_rad.max())
        radial_pres_lim = (all_pres_rad.min(), all_pres_rad.max())
        radial_vel_lim = (all_vel_rad.min(), all_vel_rad.max())
    else:
        radial_x_lim = (0, 1)
        radial_dens_lim = (0, 1)
        radial_pres_lim = (0, 1)
        radial_vel_lim = (0, 1)

    # --- Create figure with 10 rows and n_dirs columns ---
    # Rows 0: scatter; 1-3: x-slice; 4-6: y-slice; 7: radial density; 8: radial pressure; 9: radial velocity.
    fig, axes = plt.subplots(10, n_dirs, figsize=(5 * n_dirs, 18))
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
    cbar = fig.colorbar(scatters[-1], ax=axes[0, :], shrink=0.8, aspect=20, pad=0)
    cbar.set_label(physics_key)

    # --- Rows 1-3: x-slice plots ---
    sim_dens_lines = []
    analytic_dens_lines = []
    for i in range(n_dirs):
        ax = axes[1, i]
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("Density")
        ax.set_xlim(dens_x_lim)
        ax.set_ylim(dens_y_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Density")
        sim_dens_lines.append(sim_line)
        anal_line, = ax.plot([], [], 'r--', label="Analytic Density")
        analytic_dens_lines.append(anal_line)
        ax.legend(loc="upper right")
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
        anal_line, = ax.plot([], [], 'r--', label="Analytic Pressure")
        analytic_pres_lines.append(anal_line)
        ax.legend(loc="upper right")
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
        anal_line, = ax.plot([], [], 'r--', label="Analytic Velocity")
        analytic_vel_lines.append(anal_line)
        ax.legend(loc="upper right")

    # --- Rows 4-6: y-slice plots ---
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
        anal_line, = ax.plot([], [], 'r--', label="Analytic Density")
        analytic_dens_y_lines.append(anal_line)
        ax.legend(loc="upper right")
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
        anal_line, = ax.plot([], [], 'r--', label="Analytic Pressure")
        analytic_pres_y_lines.append(anal_line)
        ax.legend(loc="upper right")
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
        anal_line, = ax.plot([], [], 'r--', label="Analytic Velocity")
        analytic_vel_y_lines.append(anal_line)
        ax.legend(loc="upper right")

    # --- Rows 7-9: Radial profiles ---
    # Row 7: Density vs. radius.
    sim_radial_dens_lines = []
    # We add an extra analytic line for density.
    analytic_radial_dens_lines = []
    for i in range(n_dirs):
        ax = axes[7, i]
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Density")
        ax.set_xlim(radial_x_lim)
        ax.set_ylim(radial_dens_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Radial Density")
        sim_radial_dens_lines.append(sim_line)
        anal_line, = ax.plot([], [], linestyle='--', color='red', label="Analytic Density")
        analytic_radial_dens_lines.append(anal_line)
        ax.legend(loc="upper right")
    # Row 8: Pressure vs. radius.
    sim_radial_pres_lines = []
    analytic_radial_pres_lines = []
    for i in range(n_dirs):
        ax = axes[8, i]
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Pressure")
        ax.set_xlim(radial_x_lim)
        ax.set_ylim(radial_pres_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Radial Pressure")
        sim_radial_pres_lines.append(sim_line)
        anal_line, = ax.plot([], [], linestyle='--', color='red', label="Analytic Pressure")
        analytic_radial_pres_lines.append(anal_line)
        ax.legend(loc="upper right")
    # Row 9: Velocity vs. radius.
    sim_radial_vel_lines = []
    for i in range(n_dirs):
        ax = axes[9, i]
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Velocity")
        ax.set_xlim(radial_x_lim)
        ax.set_ylim(radial_vel_lim)
        sim_line, = ax.plot([], [], marker='o', linestyle='-', label="Sim Radial Velocity")
        sim_radial_vel_lines.append(sim_line)
        ax.legend(loc="upper right")

    n_frames = len(list_of_dataframes[0])

    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        for line in sim_dens_lines + sim_pres_lines + sim_vel_lines:
            line.set_data([], [])
        for line in sim_dens_y_lines + sim_pres_y_lines + sim_vel_y_lines:
            line.set_data([], [])
        for line in sim_radial_dens_lines + sim_radial_pres_lines + sim_radial_vel_lines:
            line.set_data([], [])
        for line in analytic_dens_lines + analytic_pres_lines + analytic_vel_lines:
            line.set_data([], [])
        for line in analytic_dens_y_lines + analytic_pres_y_lines + analytic_vel_lines:
            line.set_data([], [])
        for line in analytic_radial_dens_lines + analytic_radial_pres_lines:
            line.set_data([], [])
        return (scatters + analytic_dens_lines + analytic_pres_lines + analytic_vel_lines +
                sim_dens_lines + sim_pres_lines + sim_vel_lines +
                analytic_dens_y_lines + analytic_pres_y_lines + analytic_vel_lines +
                sim_dens_y_lines + sim_pres_y_lines + sim_vel_y_lines +
                sim_radial_dens_lines + sim_radial_pres_lines + sim_radial_vel_lines +
                analytic_radial_dens_lines + analytic_radial_pres_lines)

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

            # --- Update analytic overlays for x-slice (using shocktube analytic as before) ---
            t_current = list_of_times[i][frame_index]
            x_dense = np.linspace(dens_x_lim[0], dens_x_lim[1], 200)
            rho_a, p_a, u_a = analytic_sod_solution_1d(x_dense, t_current, gamma=1.4, x0=0.0)
            analytic_dens_lines[i].set_data(x_dense, rho_a)
            analytic_pres_lines[i].set_data(x_dense, p_a)
            analytic_vel_lines[i].set_data(x_dense, u_a)

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

            # --- Update radial profiles (rows 7-9) ---
            r = np.sqrt(df["pos_x"].values ** 2 + df["pos_y"].values ** 2)
            sorted_idx = np.argsort(r)
            r_sorted = r[sorted_idx]
            dens_rad = df["dens"].values[sorted_idx]
            pres_rad = df["pres"].values[sorted_idx]
            vel_rad = np.sqrt(df["vel_x"].values[sorted_idx] ** 2 + df["vel_y"].values[sorted_idx] ** 2)
            sim_radial_dens_lines[i].set_data(r_sorted, dens_rad)
            sim_radial_pres_lines[i].set_data(r_sorted, pres_rad)
            sim_radial_vel_lines[i].set_data(r_sorted, vel_rad)
            # --- Compute analytic Lane–Emden (n=3/2) profile ---
            if r_sorted.size > 0:
                r_max_sim = r_sorted.max()
                dens_c = dens_rad[0]  # approximate central density
                pres_c = pres_rad[0]  # approximate central pressure
                r_analytic, analytic_dens, analytic_pres = lane_emden_analytic_n32(r_max_sim, dens_c, pres_c)
                analytic_radial_dens_lines[i].set_data(r_analytic, analytic_dens)
                analytic_radial_pres_lines[i].set_data(r_analytic, analytic_pres)
            else:
                analytic_radial_dens_lines[i].set_data([], [])
                analytic_radial_pres_lines[i].set_data([], [])
        return (scatters + analytic_dens_lines + analytic_pres_lines + analytic_vel_lines +
                sim_dens_lines + sim_pres_lines + sim_vel_lines +
                analytic_dens_y_lines + analytic_pres_y_lines + analytic_vel_lines +
                sim_dens_y_lines + sim_pres_y_lines + sim_vel_y_lines +
                sim_radial_dens_lines + sim_radial_pres_lines + sim_radial_vel_lines +
                analytic_radial_dens_lines + analytic_radial_pres_lines)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout(rect=[0, 0, 0.8, 0.9])
    plt.show()
    return ani


def main():
    data_dirs = [
        "/Users/guo/OSS/sphcode/sample/lane_emden_2d/results/DISPH/lane_emden_2d/2D",
    ]
    plot_titles = [generate_title_from_dir(d) for d in data_dirs]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_2d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_2d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles)


if __name__ == "__main__":
    main()
