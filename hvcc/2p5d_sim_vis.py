#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.special import ellipk
import csv
import argparse
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'


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
    m = (2.0 * np.sqrt(R_mat * Rp_mat) / denom) ** 2
    m = np.clip(m, 0, 0.9999)
    K_mat = (2.0 / np.pi) * ellipk(m) / denom
    return K_mat


def compute_disk_potential_vectorized(R_array, Sigma_array, G):
    """
    Potential: Φ(R) = -G ∫ Σ(R') R' dR' K(R,R'),
    using a precomputed kernel matrix.
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
    Sigma = Sigma0 * np.exp(-R_array / (r_max / 5 + 1e-10))
    Sigma[0] = Sigma0

    for iteration in range(max_iter):
        Phi = compute_disk_potential_vectorized(R_array, Sigma, G)
        Phi0 = Phi[0]
        argument = Sigma0 ** (2 / 3) - (2.0 / (5 * K_poly)) * (Phi - Phi0)
        Sigma_new = np.power(np.maximum(argument, 0.0), 3 / 2)
        Sigma_new = 0.1 * Sigma_new + 0.9 * Sigma
        diff = np.linalg.norm(Sigma_new - Sigma)
        if diff < tol:
            Sigma = Sigma_new.copy()
            break
        Sigma = Sigma_new.copy()

    analytic_dens = Sigma
    analytic_pres = K_poly * np.power(Sigma, 5 / 3)
    return R_array, analytic_dens, analytic_pres


def plot_csv_analytic(ax, csv_filename, color='magenta', label='CSV'):
    """
    Reads the CSV file (with columns R, Sigma) and plots it on the given Axes.
    """
    R_vals = []
    Sigma_vals = []
    try:
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                R_vals.append(float(row[0]))
                Sigma_vals.append(float(row[1]))
        R_vals = np.array(R_vals)
        Sigma_vals = np.array(Sigma_vals)
        ax.plot(R_vals, Sigma_vals, color=color, linestyle=':', label=label)
    except Exception as e:
        print(f"Could not plot CSV analytic data: {e}")


# -------------------------
# Animation Controller for Interactive Controls
# -------------------------
class AnimationController:
    def __init__(self, fig, ani, update_func, n_frames):
        self.fig = fig
        self.ani = ani
        self.update_func = update_func
        self.n_frames = n_frames
        self.current_frame = 0
        self.paused = False
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        print("Interactive controls: space to pause/resume, left/right to step, 'r' to rewind.")

    def on_key(self, event):
        if event.key == " ":
            if self.paused:
                self.ani.event_source.start()
                self.paused = False
                print("Resumed.")
            else:
                self.ani.event_source.stop()
                self.paused = True
                print("Paused.")
        elif event.key == "right":
            self.ani.event_source.stop()
            self.current_frame = (self.current_frame + 1) % self.n_frames
            self.update_func(self.current_frame)
            self.fig.canvas.draw_idle()
            print(f"Step forward to frame {self.current_frame}.")
        elif event.key == "left":
            self.ani.event_source.stop()
            self.current_frame = (self.current_frame - 1) % self.n_frames
            self.update_func(self.current_frame)
            self.fig.canvas.draw_idle()
            print(f"Step backward to frame {self.current_frame}.")
        elif event.key == "r":
            self.ani.event_source.stop()
            self.current_frame = 0
            self.update_func(0)
            self.fig.canvas.draw_idle()
            print("Rewound to frame 0.")


# -------------------------
# 3D-only Animation (Full-screen)
# -------------------------
def animate_3d_only(list_of_dataframes, list_of_times, physics_key="dens", plot_title="3D Plot", interactive=True):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(plot_title)

    # Set axis limits from the first dataset
    pos_x_all = np.concatenate([df["pos_x"].values for df in list_of_dataframes[0]])
    pos_y_all = np.concatenate([df["pos_y"].values for df in list_of_dataframes[0]])
    pos_z_all = np.concatenate([df["pos_z"].values for df in list_of_dataframes[0]])
    ax.set_xlim(pos_x_all.min()*0.95, pos_x_all.max()*1.05)
    ax.set_ylim(pos_y_all.min()*0.95, pos_y_all.max()*1.05)
    ax.set_zlim(pos_z_all.min()*0.95, pos_z_all.max()*1.05)

    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=16)
    scatter = ax.scatter([], [], [], s=1, c=[], cmap="viridis")
    scatter_floored = ax.scatter([], [], [], s=10, color='red')

    n_frames = len(list_of_dataframes[0])

    def init():
        scatter._offsets3d = ([], [], [])
        scatter.set_array(np.array([]))
        scatter_floored._offsets3d = ([], [], [])
        time_text.set_text('')
        return scatter, scatter_floored, time_text

    def update(frame_index):
        df = list_of_dataframes[0][frame_index]
        x = df["pos_x"].values
        y = df["pos_y"].values
        z = df["pos_z"].values
        physics_values = df[physics_key].values
        ene_floored = df["ene_floored"].values
        mask = ene_floored == 1
        # Update non-floored points
        scatter._offsets3d = (x[~mask], y[~mask], z[~mask])
        scatter.set_array(physics_values[~mask])
        # Update floored points
        scatter_floored._offsets3d = (x[mask], y[mask], z[mask])
        current_time = list_of_times[0][frame_index]
        time_text.set_text(f"Time: {current_time:.4f} s")
        return scatter, scatter_floored, time_text

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=300, blit=False)
    ani.n_frames = n_frames
    if interactive:
        AnimationController(fig, ani, update, n_frames)
    return fig, ani


# -------------------------
# Full (Multi-Panel) Animation
# -------------------------
def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens",
                        plot_titles=None, csv_filename="",
                        interactive=True):
    n_dirs = len(list_of_dataframes)
    fig = plt.figure(figsize=(6 * n_dirs * 2, 10))
    gs = fig.add_gridspec(3, 2 * n_dirs, hspace=0.3, wspace=0.3)

    axes_3d = []
    axes_intersect = {"dens": [], "pres": [], "vel": [], "ene": []}

    for i in range(n_dirs):
        ax3d = fig.add_subplot(gs[0, 2 * i:2 * i + 2], projection='3d')
        axes_3d.append(ax3d)
        title = plot_titles[i] if (plot_titles and i < len(plot_titles)) else f"Dir {i + 1}"
        ax3d.set_title(title)
        pos_x_all = np.concatenate([df["pos_x"].values for df in list_of_dataframes[i]])
        pos_y_all = np.concatenate([df["pos_y"].values for df in list_of_dataframes[i]])
        pos_z_all = np.concatenate([df["pos_z"].values for df in list_of_dataframes[i]])
        ax3d.set_xlim(pos_x_all.min()*0.95, pos_x_all.max()*1.05)
        ax3d.set_ylim(pos_y_all.min()*0.95, pos_y_all.max()*1.05)
        ax3d.set_zlim(pos_z_all.min()*0.95, pos_z_all.max()*1.05)

        ax_dens = fig.add_subplot(gs[1, 2 * i])
        ax_pres = fig.add_subplot(gs[1, 2 * i + 1])
        axes_intersect["dens"].append(ax_dens)
        axes_intersect["pres"].append(ax_pres)

        ax_vel = fig.add_subplot(gs[2, 2 * i])
        ax_ene = fig.add_subplot(gs[2, 2 * i + 1])
        axes_intersect["vel"].append(ax_vel)
        axes_intersect["ene"].append(ax_ene)

    time_text = fig.text(0.5, 0.97, '', ha='center', fontsize=16)

    # Initialize line objects for each subplot
    lines_dens = []
    lines_pres = []
    lines_vel = []
    lines_energy = []
    analytic_lines_dens = []
    analytic_lines_pres = []

    for i in range(n_dirs):
        line_dens, = axes_intersect["dens"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        analytic_line_dens, = axes_intersect["dens"][i].plot([], [], linestyle='--', color='red', label='Analytic')
        lines_dens.append(line_dens)
        analytic_lines_dens.append(analytic_line_dens)
        axes_intersect["dens"][i].set_xlabel("Radius [m]")
        axes_intersect["dens"][i].set_ylabel("Density")
        axes_intersect["dens"][i].legend()

        line_pres, = axes_intersect["pres"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        analytic_line_pres, = axes_intersect["pres"][i].plot([], [], linestyle='--', color='red', label='Analytic')
        lines_pres.append(line_pres)
        analytic_lines_pres.append(analytic_line_pres)
        axes_intersect["pres"][i].set_xlabel("Radius [m]")
        axes_intersect["pres"][i].set_ylabel("Pressure")
        axes_intersect["pres"][i].legend()

        line_vel, = axes_intersect["vel"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        lines_vel.append(line_vel)
        axes_intersect["vel"][i].set_xlabel("Radius [m]")
        axes_intersect["vel"][i].set_ylabel("Velocity")
        axes_intersect["vel"][i].legend()

        line_ene, = axes_intersect["ene"][i].plot([], [], marker='o', linestyle='-', markersize=2, label='Sim')
        lines_energy.append(line_ene)
        axes_intersect["ene"][i].set_xlabel("Radius [m]")
        axes_intersect["ene"][i].set_ylabel("Energy")
        axes_intersect["ene"][i].legend()

    n_frames = len(list_of_dataframes[0])

    # For the 3D axes, create two scatter objects (for regular and ene_floored points)
    for ax in axes_3d:
        ax._scatter_obj = ax.scatter([], [], [], s=1, c=[], cmap="viridis")
        ax._scatter_floored = ax.scatter([], [], [], s=10, color='red')

    def init():
        for ax in axes_3d:
            ax._scatter_obj._offsets3d = ([], [], [])
            ax._scatter_obj.set_array(np.array([]))
            ax._scatter_floored._offsets3d = ([], [], [])
        for line in (lines_dens + lines_pres + lines_vel + lines_energy +
                     analytic_lines_dens + analytic_lines_pres):
            line.set_data([], [])
        time_text.set_text('')
        # Plot CSV analytic curves if provided
        for i in range(n_dirs):
            if csv_filename:
                plot_csv_analytic(axes_intersect["dens"][i], csv_filename, color='magenta', label='CSV')
        return [time_text] + [ax._scatter_obj for ax in axes_3d] + [ax._scatter_floored for ax in axes_3d] + \
               lines_dens + lines_pres + lines_vel + lines_energy + \
               analytic_lines_dens + analytic_lines_pres

    def update(frame_index):
        time_text.set_text(f"Time: {list_of_times[0][frame_index]:.4f} s")
        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            # Update 3D scatter objects
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            ene_floored = df["ene_floored"].values
            physics_values = df[physics_key].values
            mask = ene_floored == 1
            axes_3d[i]._scatter_obj._offsets3d = (x[~mask], y[~mask], z[~mask])
            axes_3d[i]._scatter_obj.set_array(physics_values[~mask])
            axes_3d[i]._scatter_floored._offsets3d = (x[mask], y[mask], z[mask])

            # Prepare 2D plots
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

            lines_dens[i].set_data(r_sorted, dens)
            lines_pres[i].set_data(r_sorted, pres)
            lines_vel[i].set_data(r_sorted, vel)
            lines_energy[i].set_data(r_sorted, ene)
            axes_intersect["dens"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["dens"][i].set_ylim(-0.2, dens.max())
            axes_intersect["pres"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["pres"][i].set_ylim(-0.2, pres.max())
            axes_intersect["vel"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["vel"][i].set_ylim(vel.min(), vel.max())
            axes_intersect["ene"][i].set_xlim(r_sorted.min(), r_sorted.max()*1.25)
            axes_intersect["ene"][i].set_ylim(-0.2, 300)

            if r_sorted.size > 0:
                r_max_sim = r_sorted.max()
                Sigma0_local = dens[0] if dens[0] > 0 else 1.0
                K_poly_local = pres[0] / (dens[0] ** (5 / 3)) if pres[0] > 0 else 1.0
                r_analytic, an_dens, an_pres = razor_thin_disk_analytic(r_max_sim, Sigma0_local, K_poly_local)
                analytic_lines_dens[i].set_data(r_analytic, an_dens)
                analytic_lines_pres[i].set_data(r_analytic, an_pres)
            else:
                analytic_lines_dens[i].set_data([], [])
                analytic_lines_pres[i].set_data([], [])
        return [time_text] + [ax._scatter_obj for ax in axes_3d] + [ax._scatter_floored for ax in axes_3d] + \
               lines_dens + lines_pres + lines_vel + lines_energy + \
               analytic_lines_dens + analytic_lines_pres

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=300, blit=False)
    ani.n_frames = n_frames
    if interactive:
        AnimationController(fig, ani, update, n_frames)
    return fig, ani


# -------------------------
# Main: Parse Arguments and Run Animation
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Animation script with interactive controls and recording options")
    parser.add_argument("--mode", choices=["interactive", "record"], default="record",
                        help="Run in interactive mode (default) or record a movie")
    parser.add_argument("--plot", choices=["full", "3d"], default="3d",
                        help="Select plot type: 'full' for multi-panel plot, '3d' for 3D-only full-size plot")
    parser.add_argument("--output", type=str, default="./20250319.mov",
                        help="Output filename for recorded .mov file (required in record mode)")
    args = parser.parse_args()

    # Load data (adjust the data_dirs as needed)
    data_dirs = [
        # Example: "/path/to/DISPH/razor_thin_hvcc_debug/2.5D",
        "/Users/guo/OSS/sphcode/production_sims/razor_thin_hvcc/results/DISPH/razor_thin_hvcc/2.5D",
    ]
    plot_titles = ["Dataset 1"]
    list_of_dataframes = []
    list_of_times = []
    for dpath in data_dirs:
        dfs, times = load_dataframes_3d(dpath)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)

    interactive = (args.mode == "interactive")

    if args.plot == "full":
        fig, ani = animate_multiple_3d(list_of_dataframes, list_of_times,
                                       physics_key="dens", plot_titles=plot_titles,
                                       csv_filename="../razor_thin_disk/adiabatic_razor_thin_disk_sigma.csv",
                                       interactive=interactive)
    elif args.plot == "3d":
        fig, ani = animate_3d_only(list_of_dataframes, list_of_times,
                                   physics_key="dens", plot_title="3D Full-Size Plot",
                                   interactive=interactive)

    if args.mode == "record":
        if not args.output:
            print("Error: In record mode, you must specify an output filename (e.g. --output animation.mov)")
            return
        writer = FFMpegWriter(fps=30)
        ani.save(args.output, writer=writer)
        print(f"Movie saved as {args.output}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
