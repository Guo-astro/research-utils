# plot1d.py
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from common import extract_time, load_dataframes_1d


def animate_multiple_1d(list_of_dataframes, list_of_times, plot_titles=None):
    """
    Animate 1D scatter plots (pos_x vs dens).
    """
    n_dirs = len(list_of_dataframes)
    fig, axes = plt.subplots(1, n_dirs, figsize=(5 * n_dirs, 4))
    if n_dirs == 1:
        axes = [axes]
    scatters = []
    for i, dataframes in enumerate(list_of_dataframes):
        ax = axes[i]
        pos_x_all = np.concatenate([df["pos_x"].values for df in dataframes])
        dens_all = np.concatenate([df["dens"].values for df in dataframes])
        ax.set_xlim(pos_x_all.min() * 0.95, pos_x_all.max() * 1.05)
        ax.set_ylim(dens_all.min() * 0.95, dens_all.max() * 1.05)
        ax.set_xlabel("pos_x [m]")
        ax.set_ylabel("dens [kg/m³]")
        scat = ax.scatter([], [], s=10, c="blue")
        scatters.append(scat)
        ax.set_title(plot_titles[i] if plot_titles and i < len(plot_titles) else f"Dir {i + 1}")
    n_frames = len(list_of_dataframes[0])

    def init():
        for scat in scatters:
            scat.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame_index):
        for i, dataframes in enumerate(list_of_dataframes):
            df = dataframes[frame_index]
            x = df["pos_x"].values
            y = df["dens"].values
            scatters[i].set_offsets(np.column_stack((x, y)))
        return scatters

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=100, blit=True)
    plt.show()
    return ani


def main():
    # Example usage for 1D plots.
    data_dirs = [
        "/path/to/your/2d_dir1",
        "/path/to/your/2d_dir2"
    ]
    plot_titles = ["Dataset 1", "Dataset 2"]
    list_of_dataframes = []
    list_of_times = []
    for data_path in data_dirs:
        dfs, times = load_dataframes_1d(data_path)
        list_of_dataframes.append(dfs)
        list_of_times.append(times)
    animate_multiple_1d(list_of_dataframes, list_of_times, plot_titles=plot_titles)


if __name__ == "__main__":
    main()
