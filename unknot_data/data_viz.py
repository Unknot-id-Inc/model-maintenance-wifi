import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

COLORS = list(mcolors.CSS4_COLORS.keys())
random.seed(5)
random.shuffle(COLORS)


x_axis, y_axis, z_axis = (
    np.array([[1.0, 0.0, 0.0]]),
    np.array([[0.0, 1.0, 0.0]]),
    np.array([[0.0, 0.0, 1.0]]),
)

import numpy as np


def ax_array(
    rows=1, cols=1, size=(3, 6), suptitle="", supsize=18, return_fig=False, sharex=False
):
    fig, axs = plt.subplots(rows, cols, figsize=(size[1], size[0]), sharex=sharex)
    fig.suptitle(suptitle, fontsize=supsize)

    if rows == 1 and cols == 1:
        if return_fig:
            return fig, np.array([[axs]])
        return np.array([[axs]])
    elif rows == 1:
        if return_fig:
            return fig, np.array([axs])
        return np.array([axs])
    elif cols == 1:
        if return_fig:
            return fig, np.array([axs]).T
        return np.array([axs]).T
    else:
        if return_fig:
            return fig, axs
        return axs


def ax3d_array(rows=1, cols=1, size=(3, 6), suptitle=""):
    fig, axs = plt.subplots(
        rows, cols, figsize=(size[1], size[0]), subplot_kw=dict(projection="3d")
    )
    fig.suptitle(suptitle)

    if rows == 1 and cols == 1:
        return np.array([[axs]])
    elif rows == 1:
        return np.array([axs])
    elif cols == 1:
        return np.array([axs]).T
    else:
        return axs


def plot_3d(ax, data, **kwargs):
    for i in range(data.shape[1]):
        ax.plot(data[:, i], **kwargs)


def plot_lines_3d(ax, origins, colors=None, **kwargs):
    ax.plot(
        origins[:, 0].squeeze(),
        origins[:, 1].squeeze(),
        origins[:, 2].squeeze(),
        **kwargs,
    )


def plot_vectors_3d(
    ax,
    origins,
    *vectors,
    granularity=None,
    length=None,
    norm=True,
    colors=None,
    **kwargs,
):
    if origins.shape[0] > 1:
        xrange = (origins[:, 0].min(), origins[:, 0].max())
        yrange = (origins[:, 1].min(), origins[:, 1].max())
        zrange = (origins[:, 2].min(), origins[:, 2].max())
    else:
        vmax = max([np.linalg.norm(x, axis=-1) for x in vectors])
        xrange = (-vmax, vmax)
        yrange = (-vmax, vmax)
        zrange = (-vmax, vmax)
        length = 1

    max_range = max([x[1] - x[0] for x in [xrange, yrange, zrange]])
    ax.set_xlim(np.mean(xrange) - max_range / 2, np.mean(xrange) + max_range / 2)
    ax.set_ylim(np.mean(yrange) - max_range / 2, np.mean(yrange) + max_range / 2)
    ax.set_zlim(np.mean(zrange) - max_range / 2, np.mean(zrange) + max_range / 2)

    if length == None:
        length = 0.1 * max_range

    if origins.shape[0] > 1:
        if granularity == None:
            granularity = int(origins.shape[0] / 100)
    else:
        granularity = 1

    points = np.arange(0, origins.shape[0], granularity)

    x, y, z = origins[points, 0], origins[points, 1], origins[points, 2]

    ax.plot(
        origins[points, 0].squeeze(),
        origins[points, 1].squeeze(),
        origins[points, 2].squeeze(),
    )

    if len(vectors) > 0:
        if norm:
            vectors = vectors / np.linalg.norm(vectors, axis=2, keepdims=True)

        for i, vector in enumerate(vectors):
            u = vector[points, 0]
            v = vector[points, 1]
            w = vector[points, 2]

            # print(x, y, z, u, v, w, length)
            if i > 0 and "label" in kwargs:
                kwargs.pop("label")

            ax.quiver(
                x,
                y,
                z,
                u,
                v,
                w,
                length=length,
                arrow_length_ratio=0.2,
                color=[colors[i]],
                **kwargs,
            )


from unknot.utils.data_viz import ax_array


class AxisGrid:
    def __init__(
        self, N: int = 1, n_cols: int = 1, size: tuple[int, int] = (8, 8), **kwargs
    ):
        self.N = N
        self.n_cols = n_cols
        self.tile_size = size

        self.n_rows = (
            N // self.n_cols if N % self.n_cols == 0 else (N // self.n_cols) + 1
        )
        self.fig, self._axs = ax_array(
            self.n_rows,
            self.n_cols,
            size=(self.tile_size[0] * self.n_rows, self.tile_size[1] * self.n_cols),
            return_fig=True,
            **kwargs,
        )

    def __getitem__(self, i: int):
        return self._axs[i % self.n_rows, i // self.n_rows]

    def __setitem__(self, key: int, value):
        self._axs[key % self.n_rows, key // self.n_rows] = value

    def __iter__(self):
        for i in range(self.N):
            yield self._axs[i % self.n_rows, i // self.n_rows]

    def plot_xy(self, i, poses, **kwargs):
        self[i].plot(
            [data.x for data in poses],
            [data.y for data in poses],
            **kwargs,
        )

    def scatter_xy(self, i, poses, **kwargs):
        self[i].scatter(
            [data.x for data in poses],
            [data.y for data in poses],
            **kwargs,
        )

    def plot_geo(self, i, poses, **kwargs):
        self[i].plot(
            [data.longitude for data in poses],
            [data.latitude for data in poses],
            **kwargs,
        )

    def scatter_geo(self, i, poses, **kwargs):
        self[i].scatter(
            [data.longitude for data in poses],
            [data.latitude for data in poses],
            **kwargs,
        )

    def plot_timestamped_fn(self, i, timestamped_data, fn, **kwargs):
        self[i].plot(
            [data.timestamp for data in timestamped_data],
            [fn(data) for data in timestamped_data],
            **kwargs,
        )

    def plot_magnitudes(self, i, magnitudes, **kwargs):
        self[i].plot(
            [data.timestamp for data in magnitudes],
            [data.magnitude for data in magnitudes],
            **kwargs,
        )

    def scatter_magnitudes(self, i, magnitudes, **kwargs):
        self[i].scatter(
            [data.timestamp for data in magnitudes],
            [data.magnitude for data in magnitudes],
            **kwargs,
        )

    def plot_vectors(self, i, vector_datas, **kwargs):
        self[i].plot(
            [data.timestamp for data in vector_datas],
            [data.V[0] for data in vector_datas],
            **kwargs,
        )
        self[i].plot(
            [data.timestamp for data in vector_datas],
            [data.V[1] for data in vector_datas],
            **kwargs,
        )
        self[i].plot(
            [data.timestamp for data in vector_datas],
            [data.V[2] for data in vector_datas],
            **kwargs,
        )

    def plot_inertials(self, i, inertial_datas, **kwargs):
        self[i].plot(
            [data.timestamp for data in inertial_datas],
            [data.x for data in inertial_datas],
            **kwargs,
        )
        self[i].plot(
            [data.timestamp for data in inertial_datas],
            [data.y for data in inertial_datas],
            **kwargs,
        )
        self[i].plot(
            [data.timestamp for data in inertial_datas],
            [data.z for data in inertial_datas],
            **kwargs,
        )

    def scatter_inertials(self, i, inertial_datas, **kwargs):
        self[i].scatter(
            [data.timestamp for data in inertial_datas],
            [data.x for data in inertial_datas],
            **kwargs,
        )
        self[i].scatter(
            [data.timestamp for data in inertial_datas],
            [data.y for data in inertial_datas],
            **kwargs,
        )
        self[i].scatter(
            [data.timestamp for data in inertial_datas],
            [data.z for data in inertial_datas],
            **kwargs,
        )


from matplotlib.colors import hsv_to_rgb


def color_shades(num_shades: int, saturation=1.0, value=1.0, hue="green"):
    assert hue in ["green", "red"]

    if hue == "green":
        hrange = (0.2, 0.5)
    elif hue == "red":
        hrange = (0.0, 0.15)
    else:
        raise KeyError

    hues = np.linspace(*hrange, num_shades)
    color_shades = [hsv_to_rgb([h, saturation, value]) for h in hues]
    return color_shades
