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
    """
    Animate 3D scatter + radial profiles, overlaying:
      1) On-the-fly analytic solution via razor_thin_disk_analytic
      2) The CSV solution from adiabatic_razor_thin_disk_sigma.csv (if present)
    """
    n_dirs = len(list_of_dataframes)
    fig = plt.figure(figsize=(5*n_dirs, 15))
    axes_3d = []

    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=16)

    lines_dens = []
    lines_pres = []
    lines_vel = []
    lines_energy = []
    analytic_lines_dens = []
    analytic_lines_pres = []

    # 3D scatter subplots (top row)
    for i, dataframes in enumerate(list_of_dataframes):
        ax3d = fig.add_subplot(5, n_dirs, i+1, projection="3d")
        axes_3d.append(ax3d)
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
        pos_z_all = np.concatenate([df["pos_z"].values for df in dataframes])
        ax3d.set_xlim(pos_x_all.min()*0.95, pos_x_all.max()*1.05)
        ax3d.set_ylim(pos_y_all.min()*0.95, pos_y_all.max()*1.05)
        ax3d.set_zlim(pos_z_all.min()*0.95, pos_z_all.max()*1.05)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax3d.scatter([], [], [], s=1, c=[], cmap="viridis",
                            vmin=physics_all.min(), vmax=physics_all.max())
        title = plot_titles[i] if (plot_titles and i<len(plot_titles)) else f"Dir {i+1}"
        ax3d.set_title(title)
        ax3d._scatter_obj = scat

    # radial subplots for density, pressure, velocity, energy
    radial_vars = [("dens","Density",True),
                   ("pres","Pressure",True),
                   ("vel","Velocity",False),
                   ("ene","Energy",False)]
    axes_intersect = {}
    for var,ylabel,_ in radial_vars:
        axes_intersect[var] = []
    row_map = {"dens":1, "pres":2, "vel":3, "ene":4}

    for var,ylabel,analytic_avail in radial_vars:
        row = row_map[var]
        for i in range(n_dirs):
            ax = fig.add_subplot(5,n_dirs,row*n_dirs + i+1)
            ax.set_xlabel("Radius [m]")
            ax.set_ylabel(ylabel)
            line_sim, = ax.plot([],[], marker='o', linestyle='-', markersize=2, label='Sim')
            line_analytic, = ax.plot([],[], linestyle='--', color='red', label='Analytic')
            axes_intersect[var].append(ax)
            if var=="dens":
                lines_dens.append(line_sim)
                analytic_lines_dens.append(line_analytic)
            elif var=="pres":
                lines_pres.append(line_sim)
                analytic_lines_pres.append(line_analytic)
            elif var=="vel":
                lines_vel.append(line_sim)
            elif var=="ene":
                lines_energy.append(line_sim)
            ax.legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        # Clear scatter
        for ax in axes_3d:
            ax._scatter_obj._offsets3d = (np.array([]),np.array([]),np.array([]))
        # Clear lines
        for line in (lines_dens + lines_pres + lines_vel + lines_energy +
                     analytic_lines_dens + analytic_lines_pres):
            line.set_data([], [])
        time_text.set_text('')

        # ----------------------------------------------------------------
        # Plot the CSV-based analytic solution as a static overlay
        # ONLY on the 'dens' subplots, for demonstration.
        # If you want to show it on 'pres' as well, repeat similarly.
        # ----------------------------------------------------------------
        for i in range(n_dirs):
            ax_dens = axes_intersect["dens"][i]
            plot_csv_analytic(ax_dens, csv_filename,
                              color='magenta', label='CSV')

        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel + lines_energy +
                analytic_lines_dens + analytic_lines_pres + [time_text])

    def update(frame_index):
        artists = []
        current_time = list_of_times[0][frame_index]
        time_text.set_text(f"Time: {current_time:.4f} s")
        artists.append(time_text)

        for i,(dataframes,times) in enumerate(zip(list_of_dataframes,list_of_times)):
            df = dataframes[frame_index]
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scat = axes_3d[i]._scatter_obj
            scat._offsets3d = (x, y, z)
            scat.set_array(df[physics_key].values)
            artists.append(scat)

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

            # Update lines for simulation
            lines_dens[i].set_data(r_sorted, dens)
            lines_pres[i].set_data(r_sorted, pres)
            lines_vel[i].set_data(r_sorted, vel)
            lines_energy[i].set_data(r_sorted, ene)

            # Adjust axes
            axes_intersect["dens"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["dens"][i].set_ylim(dens.min(), dens.max())
            axes_intersect["pres"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["pres"][i].set_ylim(pres.min(), pres.max())
            axes_intersect["vel"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["vel"][i].set_ylim(vel.min(), vel.max())
            axes_intersect["ene"][i].set_xlim(r_sorted.min(), r_sorted.max())
            axes_intersect["ene"][i].set_ylim(ene.min(), ene.max())

            # Recompute an on-the-fly analytic solution
            if r_sorted.size>0:
                r_max_sim = r_sorted.max()
                Sigma0_local = dens[0] if dens[0]>0 else 1.0
                if pres[0]>0:
                    K_poly_local = pres[0]/(dens[0]**(5/3))
                else:
                    K_poly_local = 1.0
                r_analytic, an_dens, an_pres = razor_thin_disk_analytic(r_max_sim,
                                                                        Sigma0_local,
                                                                        K_poly_local)
                analytic_lines_dens[i].set_data(r_analytic, an_dens)
                analytic_lines_pres[i].set_data(r_analytic, an_pres)
            else:
                analytic_lines_dens[i].set_data([],[])
                analytic_lines_pres[i].set_data([],[])

            artists.extend([lines_dens[i], lines_pres[i], lines_vel[i], lines_energy[i],
                            analytic_lines_dens[i], analytic_lines_pres[i]])
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=300, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def main():
    data_dirs = [
        "/Users/guo/OSS/sphcode/production_sims/test_razor_thin_blast_wave/results/DISPH/test_razor_thin_blast_wave/3D",
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
                        csv_filename="adiabatic_razor_thin_disk_sigma.csv")

if __name__=="__main__":
    main()
