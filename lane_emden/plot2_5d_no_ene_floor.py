#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import ellipk
import csv

# Adjust or replace this with your real data-loading function
from vis.common import load_dataframes_3d

G = 1.0  # gravitational constant

def vectorized_kernel_matrix(R_array):
    """
    Compute the kernel matrix K(R, Rp) for all pairs (R, Rp) using vectorized ops.
    K(R, Rp) = (2/π) * ellipk(m) / (R + Rp),  m=[2√(R*Rp)/(R+Rp)]²
    """
    R_mat, Rp_mat = np.meshgrid(R_array, R_array, indexing='ij')
    denom = R_mat + Rp_mat
    denom[denom < 1e-8] = 1e-8
    m = (2.0*np.sqrt(R_mat*Rp_mat)/denom)**2
    m = np.clip(m, 0, 0.9999)
    K_mat = (2.0/np.pi)*ellipk(m)/denom
    return K_mat

def compute_disk_potential_vectorized(R_array, Sigma_array, G):
    """
    Potential: Φ(R) = -G ∫ Σ(R') R' dR' K(R,R'),
    using a precomputed kernel matrix (faster O(N^2) but vectorized).
    """
    K_mat = vectorized_kernel_matrix(R_array)
    integrand = Sigma_array * R_array
    Phi = -G * np.trapezoid(K_mat * integrand, R_array, axis=1)
    return Phi

def razor_thin_disk_analytic(r_max, Sigma0, K_poly, num_steps=200, max_iter=100, tol=1e-6):
    """
    Solve the 2D disk equilibrium for γ=5/3 using a vectorized approach.
    Σ(R)^(2/3) = Σ0^(2/3) - [2/(5*K_poly)] [Φ(R) - Φ(0)]
    """
    R_array = np.linspace(0, r_max, num_steps)
    # Exponential initial guess
    Sigma = Sigma0 * np.exp(-R_array/(r_max/5 + 1e-10))
    Sigma[0] = Sigma0

    for iteration in range(max_iter):
        Phi = compute_disk_potential_vectorized(R_array, Sigma, G)
        Phi0 = Phi[0]
        argument = Sigma0**(2/3) - (2.0/(5*K_poly))*(Phi - Phi0)
        Sigma_new = np.power(np.maximum(argument, 0.0), 3/2)
        # damping
        Sigma_new = 0.1*Sigma_new + 0.9*Sigma
        diff = np.linalg.norm(Sigma_new - Sigma)
        if diff < tol:
            Sigma = Sigma_new.copy()
            break
        Sigma = Sigma_new.copy()

    analytic_dens = Sigma
    analytic_pres = K_poly * np.power(Sigma, 5/3)
    return R_array, analytic_dens, analytic_pres

# ------------------------------------------------------------------------
# NEW FUNCTION: Plot the CSV data from adiabatic_razor_thin_disk_sigma.csv
# ------------------------------------------------------------------------
def plot_csv_analytic(ax, csv_filename, color='magenta', label='CSV'):
    """
    Reads the CSV file (with columns R, Sigma) and plots it on the given Axes.
    This is a static overlay, so it won't animate with time.
    """
    R_vals = []
    Sigma_vals = []
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 2:
                continue
            R_str, Sigma_str = row[0], row[1]
            R_vals.append(float(R_str))
            Sigma_vals.append(float(Sigma_str))
    R_vals = np.array(R_vals)
    Sigma_vals = np.array(Sigma_vals)
    ax.plot(R_vals, Sigma_vals, color=color, linestyle=':', label=label)

def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens",
                        plot_titles=None, csv_filename=""):
    n_dirs = len(list_of_dataframes)
    # Create a figure with 3 rows and 2 columns per dataset.
    # For multiple datasets, we arrange 2*n_dirs columns.
    fig = plt.figure(figsize=(6 * n_dirs * 2, 10))  # adjust as needed
    gs = fig.add_gridspec(3, 2*n_dirs, hspace=0.3, wspace=0.3)

    # Prepare lists for axes.
    axes_3d = []
    axes_intersect = {"dens": [], "pres": [], "vel": [], "ene": []}

    # For each dataset, create a 3D scatter subplot (top row) that spans two columns,
    # and then create the radial subplots in the second and third rows.
    for i in range(n_dirs):
        # Top row: 3D scatter spanning two columns.
        ax3d = fig.add_subplot(gs[0, 2*i:2*i+2], projection='3d')
        axes_3d.append(ax3d)
        title = plot_titles[i] if (plot_titles and i < len(plot_titles)) else f"Dir {i+1}"
        ax3d.set_title(title)
        # Auto-estimate the axis limits using all frames from this dataset.
        pos_x_all = np.concatenate([df["pos_x"].values for df in list_of_dataframes[i]])
        pos_y_all = np.concatenate([df["pos_y"].values for df in list_of_dataframes[i]])
        pos_z_all = np.concatenate([df["pos_z"].values for df in list_of_dataframes[i]])
        ax3d.set_xlim(pos_x_all.min() * 0.95, pos_x_all.max() * 1.05)
        ax3d.set_ylim(pos_y_all.min() * 0.95, pos_y_all.max() * 1.05)
        ax3d.set_zlim(pos_z_all.min() * 0.95, pos_z_all.max() * 1.05)
        # Middle row: density (left) and pressure (right)
        ax_dens = fig.add_subplot(gs[1, 2*i])
        ax_pres = fig.add_subplot(gs[1, 2*i+1])
        axes_intersect["dens"].append(ax_dens)
        axes_intersect["pres"].append(ax_pres)

        # Bottom row: velocity (left) and energy (right)
        ax_vel = fig.add_subplot(gs[2, 2*i])
        ax_ene = fig.add_subplot(gs[2, 2*i+1])
        axes_intersect["vel"].append(ax_vel)
        axes_intersect["ene"].append(ax_ene)

    # Add a common time label at the top.
    time_text = fig.text(0.5, 0.97, '', ha='center', fontsize=16)

    # Prepare line objects for simulation and analytic curves.
    lines_dens = []
    lines_pres = []
    lines_vel = []
    lines_energy = []
    analytic_lines_dens = []
    analytic_lines_pres = []

    for i in range(n_dirs):
        # Density subplot.
        line_dens, = axes_intersect["dens"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        analytic_line_dens, = axes_intersect["dens"][i].plot([], [], linestyle='--', color='red', label='Analytic')
        lines_dens.append(line_dens)
        analytic_lines_dens.append(analytic_line_dens)
        axes_intersect["dens"][i].set_xlabel("Radius [m]")
        axes_intersect["dens"][i].set_ylabel("Density")
        axes_intersect["dens"][i].legend()

        # Pressure subplot.
        line_pres, = axes_intersect["pres"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        analytic_line_pres, = axes_intersect["pres"][i].plot([], [], linestyle='--', color='red', label='Analytic')
        lines_pres.append(line_pres)
        analytic_lines_pres.append(analytic_line_pres)
        axes_intersect["pres"][i].set_xlabel("Radius [m]")
        axes_intersect["pres"][i].set_ylabel("Pressure")
        axes_intersect["pres"][i].legend()

        # Velocity subplot.
        line_vel, = axes_intersect["vel"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        lines_vel.append(line_vel)
        axes_intersect["vel"][i].set_xlabel("Radius [m]")
        axes_intersect["vel"][i].set_ylabel("Velocity")
        axes_intersect["vel"][i].legend()

        # Energy subplot.
        line_ene, = axes_intersect["ene"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        lines_energy.append(line_ene)
        axes_intersect["ene"][i].set_xlabel("Radius [m]")
        axes_intersect["ene"][i].set_ylabel("Energy")
        axes_intersect["ene"][i].legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        # Initialize the 3D scatter plots.
        for ax in axes_3d:
            ax._scatter_obj = ax.scatter([], [], [], s=1, c=[], cmap="viridis")
        # Clear all simulation and analytic lines.
        for line in (lines_dens + lines_pres + lines_vel + lines_energy +
                     analytic_lines_dens + analytic_lines_pres):
            line.set_data([], [])
        time_text.set_text('')

        # Optionally overlay the CSV-based analytic solution on the density plots.
        for i in range(n_dirs):
            plot_csv_analytic(axes_intersect["dens"][i], csv_filename,
                              color='magenta', label='CSV')
        return [time_text] + [ax._scatter_obj for ax in axes_3d] + \
               lines_dens + lines_pres + lines_vel + lines_energy + \
               analytic_lines_dens + analytic_lines_pres

    def update(frame_index):
        artists = []
        current_time = list_of_times[0][frame_index]
        time_text.set_text(f"Time: {current_time:.4f} s")
        artists.append(time_text)

        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scatter_obj = axes_3d[i]._scatter_obj
            scatter_obj._offsets3d = (x, y, z)
            scatter_obj.set_array(df[physics_key].values)
            artists.append(scatter_obj)

            r = np.sqrt(x**2 + y**2 + z**2)
            sort_idx = np.argsort(r)
            r_sorted = r[sort_idx]
            dens = df["dens"].values[sort_idx]
            pres = df["pres"].values[sort_idx]
            if "vel_z" in df.columns:
                vel = np.sqrt(df["vel_x"].values[sort_idx]**2 +
                              df["vel_y"].values[sort_idx]**2 +
                              df["vel_z"].values[sort_idx]**2)
            else:
                vel = np.sqrt(df["vel_x"].values[sort_idx]**2 +
                              df["vel_y"].values[sort_idx]**2)
            if "ene" in df.columns:
                ene = df["ene"].values[sort_idx]
            else:
                ene = np.full_like(r_sorted, np.nan)

            # Update simulation curves.
            lines_dens[i].set_data(r_sorted, dens)
            lines_pres[i].set_data(r_sorted, pres)
            lines_vel[i].set_data(r_sorted, vel)
            lines_energy[i].set_data(r_sorted, ene)

            # Adjust axes limits.
            axes_intersect["dens"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["dens"][i].set_ylim(-0.2, dens.max())
            axes_intersect["pres"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["pres"][i].set_ylim(-0.2, pres.max())
            axes_intersect["vel"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["vel"][i].set_ylim(vel.min(), vel.max())
            axes_intersect["ene"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["ene"][i].set_ylim(-0.2, 500)

            # Compute an on-the-fly analytic solution.
            if r_sorted.size > 0:
                r_max_sim = r_sorted.max()
                Sigma0_local = dens[0] if dens[0] > 0 else 1.0
                K_poly_local = pres[0] / (dens[0]**(5/3)) if pres[0] > 0 else 1.0
                r_analytic, an_dens, an_pres = razor_thin_disk_analytic(r_max_sim,
                                                                        Sigma0_local,
                                                                        K_poly_local)
                analytic_lines_dens[i].set_data(r_analytic, an_dens)
                analytic_lines_pres[i].set_data(r_analytic, an_pres)
            else:
                analytic_lines_dens[i].set_data([], [])
                analytic_lines_pres[i].set_data([], [])

            artists.extend([lines_dens[i], lines_pres[i], lines_vel[i], lines_energy[i],
                            analytic_lines_dens[i], analytic_lines_pres[i]])
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=300, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def main():
    data_dirs = [
        "/Users/guo/OSS/sphcode/production_sims/razor_thin_hvcc/results/DISPH/razor_thin_hvcc/2.5D",
        # "/Users/guo/OSS/sphcode/sample/thin_slice_poly_2_5d_relax/results/DISPH/thin_slice_poly_2_5d_relax/2.5D",
        # "/Users/guo/OSS/sphcode/sample/thin_slice_poly_2_5d_relax/results/DISPH/thin_slice_poly_2_5d_relax/3D",
        # "/Users/guo/OSS/sphcode/production_sims/razor_thin_sg_relaxation/results/DISPH/razor_thin_sg_relaxation/2.5D",

    ]
    plot_titles = ["Dataset 1"]
    list_of_dataframes = []
    list_of_times = []
    for dpath in data_dirs:
        dfs, times = load_dataframes_3d(dpath)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)

    # Pass the CSV filename so we can overlay it
    animate_multiple_3d(list_of_dataframes, list_of_times,
                        physics_key="dens", plot_titles=plot_titles,
                        csv_filename="../razor_thin_disk/adiabatic_razor_thin_disk_sigma.csv")

if __name__=="__main__":
    main()
