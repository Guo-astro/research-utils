from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from common import load_dataframes_3d
from sod_analytic.shocktube_analytic_2p5d import analytic_sod_solution_2p5d


def animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=None,
                        dz_sim=1.0, h_left=0.00325, h_right=0.007, use_simulated_densities=True):
    """
    Animate 3D scatter plots with 2D intersections, overlaying the analytic shock-tube solution.

    Parameters:
      dz_sim: Effective layer thickness (1.0).
      h_left: Smoothing length for left region (0.00325).
      h_right: Smoothing length for right region (0.007).
      use_simulated_densities: If True, scale analytic rho and P to match SPH rho_sim; if False, use rho_phys.
    """
    n_dirs = len(list_of_dataframes)
    fig = plt.figure(figsize=(5 * n_dirs, 12))
    axes_3d = []
    lines_dens, lines_pres, lines_vel = [], [], []
    analytic_lines_dens, analytic_lines_pres, analytic_lines_vel = [], [], []

    # 3D scatter plots
    for i, dataframes in enumerate(list_of_dataframes):
        ax3d = fig.add_subplot(4, n_dirs, i + 1, projection="3d")
        axes_3d.append(ax3d)
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        pos_y_all = np.concatenate([df["pos_y"].values for df in dataframes])
        pos_z_all = np.concatenate([df["pos_z"].values for df in dataframes])
        ax3d.set_xlim(-1, 1)
        ax3d.set_ylim(0, 0.5)
        ax3d.set_zlim(pos_z_all.min() * 0.95, pos_z_all.max() * 1.05)
        ax3d.set_xlabel("pos_x [m]")
        ax3d.set_ylabel("pos_y [m]")
        ax3d.set_zlabel("pos_z [m]")
        physics_all = np.concatenate([df[physics_key].values for df in dataframes])
        scat = ax3d.scatter([], [], [], s=3, c=[], cmap="viridis",
                            vmin=physics_all.min(), vmax=physics_all.max())
        title = (plot_titles[i] + " (3D)") if (plot_titles and i < len(plot_titles)) else f"Dir {i + 1}"
        ax3d.set_title(title)
        ax3d._scatter_obj = scat

    # 2D intersection plots
    axes_intersect = {}
    for var, row, ylabel in zip(["dens", "pres", "vel"], [1, 2, 3], ["Density", "Pressure", "Velocity"]):
        axes_intersect[var] = []
        for i in range(n_dirs):
            ax = fig.add_subplot(4, n_dirs, n_dirs * row + i + 1)
            ax.set_xlabel("pos_x [m]")
            ax.set_ylabel(ylabel)
            line_sim, = ax.plot([], [], marker='o', linestyle='-', label='Sim')
            line_analytic, = ax.plot([], [], linestyle='--', color='red', label='Analytic')
            axes_intersect[var].append(ax)
            if var == "dens":
                lines_dens.append(line_sim)
                analytic_lines_dens.append(line_analytic)
            elif var == "pres":
                lines_pres.append(line_sim)
                analytic_lines_pres.append(line_analytic)
            elif var == "vel":
                lines_vel.append(line_sim)
                analytic_lines_vel.append(line_analytic)
            ax.legend()

    n_frames = len(list_of_dataframes[0])

    def init():
        for ax3d in axes_3d:
            ax3d._scatter_obj._offsets3d = (np.array([]), np.array([]), np.array([]))
        for line in (lines_dens + lines_pres + lines_vel +
                     analytic_lines_dens + analytic_lines_pres + analytic_lines_vel):
            line.set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    def update(frame_index):
        for i, (dataframes, times) in enumerate(zip(list_of_dataframes, list_of_times)):
            df = dataframes[frame_index]
            t = times[frame_index]

            # 3D scatter
            x = df["pos_x"].values
            y = df["pos_y"].values
            z = df["pos_z"].values
            scat = axes_3d[i]._scatter_obj
            scat._offsets3d = (x, y, z)
            scat.set_array(df[physics_key].values)

            # 2D intersection
            pos_y = df["pos_y"].values
            median_y = np.median(pos_y)
            tol = 0.01
            mask = np.abs(pos_y - median_y) < tol
            if np.any(mask):
                x_slice = df["pos_x"].values[mask]
                sorted_idx = np.argsort(x_slice)
                x_sorted = x_slice[sorted_idx]
                dens_slice = df["dens"].values[mask][sorted_idx]
                pres_slice = df["pres"].values[mask][sorted_idx]
                vel_slice = df["vel_x"].values[mask][sorted_idx]

                lines_dens[i].set_data(x_sorted, dens_slice)
                lines_pres[i].set_data(x_sorted, pres_slice)
                lines_vel[i].set_data(x_sorted, vel_slice)

                # Analytic solution
                x_grid = np.linspace(-1, 1, 500)
                rho_analytic, pres_analytic, u_analytic = analytic_sod_solution_2p5d(
                    x_grid, t, gamma=1.4, x0=0.0, dz=dz_sim, h_left=h_left, h_right=h_right)

                # Adjust scaling based on SPH output type
                if not use_simulated_densities:
                    F_left = dz_sim * (0.70 / h_left)
                    F_right = dz_sim * (0.70 / h_right)
                    scaling = np.where(x_grid < 0, F_left, F_right)  # Use initial discontinuity at t=0
                    rho_analytic /= scaling
                    pres_analytic /= scaling  # Velocity remains unscaled

                analytic_lines_dens[i].set_data(x_grid, rho_analytic)
                analytic_lines_pres[i].set_data(x_grid, pres_analytic)
                analytic_lines_vel[i].set_data(x_grid, u_analytic)

                # Debug output for first frame and a later frame
                if i == 0 and (frame_index == 2 or frame_index == n_frames // 2):
                    print(f"Frame {frame_index}, t={t:.3f}:")
                    print(f"SPH dens: {dens_slice[:5]}, pres: {pres_slice[:5]}, vel: {vel_slice[:5]}")
                    idx_mid = len(x_grid) // 2
                    print(f"Analytic dens: {rho_analytic[idx_mid - 2:idx_mid + 3]}, "
                          f"pres: {pres_analytic[idx_mid - 2:idx_mid + 3]}, vel: {u_analytic[idx_mid - 2:idx_mid + 3]}")

                # Adjust y-limits
                for ax, sim_data, ana_data in zip(
                        [axes_intersect["dens"][i], axes_intersect["pres"][i], axes_intersect["vel"][i]],
                        [dens_slice, pres_slice, vel_slice],
                        [rho_analytic, pres_analytic, u_analytic]):
                    ax.set_xlim(-1, 1)
                    combined_min = min(sim_data.min(), ana_data.min())
                    combined_max = max(sim_data.max(), ana_data.max())
                    ax.set_ylim(combined_min * 0.95, combined_max * 1.05)
            else:
                for line in [lines_dens[i], lines_pres[i], lines_vel[i],
                             analytic_lines_dens[i], analytic_lines_pres[i], analytic_lines_vel[i]]:
                    line.set_data([], [])
        return ([ax._scatter_obj for ax in axes_3d] +
                lines_dens + lines_pres + lines_vel +
                analytic_lines_dens + analytic_lines_pres + analytic_lines_vel)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


def main():
    data_dirs = ["/Users/guo/OSS/sphcode/results"]
    plot_titles = ["Dataset 1"]
    list_of_dataframes, list_of_times = [], []
    for data_path in data_dirs:
        dfs, times = load_dataframes_3d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)

    # Toggle this based on SPH output: True for rho_sim, False for rho_phys
    animate_multiple_3d(list_of_dataframes, list_of_times, physics_key="dens", plot_titles=plot_titles,
                        dz_sim=1.0, h_left=0.00375, h_right=0.009, use_simulated_densities=False)


if __name__ == "__main__":
    main()