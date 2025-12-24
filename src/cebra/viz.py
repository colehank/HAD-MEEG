import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import colors as mcolors
import plotly.graph_objects as go
from typing import Literal
import matplotlib as mpl


def plot_cebra_label(
    X: np.ndarray,
    Y: np.ndarray,
    cls_label: dict[str, np.int_],
    dtype: str = "meg",
    debug: bool = False,
    fontsize: int = 12,
    figsize: tuple[float, float] = (6, 3),
    cmap_time: str = "PuBuGn_r",
    cmap_signal: str = "Greys",
) -> plt.Figure:
    """Plot the input data and labels."""
    if debug:
        X = X[:1000, :]
        Y = Y[:1000, :]
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].imshow(X.T, aspect="auto", cmap=cmap_signal)
    axes[0].set_ylabel(f"{dtype.upper()}\nchannels", fontsize=fontsize)
    axes[0].set_yticks(np.linspace(0, X.shape[1] - 1, 5, dtype=int))
    axes[0].tick_params(axis="both", labelsize=fontsize - 2)
    axes[0].set_xticks([])
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)
    # axes[0].spines['left'].set_visible(False)

    time = Y[:, 0]  # continues time
    label = Y[:, 1]  # discrete class
    sc = axes[1].scatter(
        np.arange(Y.shape[0]),
        label,
        c=time,
        s=2,
        cmap=cmap_time,
    )
    ticklabels = [{v: k for k, v in cls_label.items()}[k] for k in np.unique(label)]
    cls_label = {k: v for k, v in cls_label.items() if k in ticklabels}
    axes[1].set_ylabel("Class labels", fontsize=fontsize)
    # axes[1].set_xlabel('Data labels')
    axes[1].set_xticks([])
    axes[1].set_yticks(np.arange(len(cls_label)))
    axes[1].set_yticklabels(ticklabels)
    axes[1].tick_params(axis="both", labelsize=fontsize - 2)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)

    for ax in axes:
        label = ax.yaxis.get_label()
        label.set_rotation_mode("anchor")
        label.set_verticalalignment("top")
        # label.set_verticalalignment('bottom')

        ax.yaxis.set_label_coords(-0.22, 0.5)
    cax = axes[1].inset_axes([0.8, 0.95, 0.15, 0.05])

    # 把 colorbar 画在 cax 里，而不是单独占一行
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Time labels (s)", fontsize=fontsize)
    sc.set_clim(-0.1, 2)
    cbar.set_ticks([-0.1, 1, 2])
    cbar.set_ticklabels(["-0.1", "1", "2"])
    cbar.ax.tick_params(labelsize=fontsize - 2)
    return fig, axes


def plot_cebra_label_meeg(
    X_meg: np.ndarray,
    Y_meg: np.ndarray,
    X_eeg: np.ndarray,
    Y_eeg: np.ndarray,
    cls_label: dict[str, np.int_],
    debug: bool = False,
    figsize: tuple[float, float] = (6, 4.5),
    fontsize: int = 12,
    cmap_time: str = "PuBuGn_r",
    cmap_signal: str = "Greys",
) -> plt.Figure:
    """Plot the input data and labels for MEG and EEG."""
    if debug:
        X_meg = X_meg[:1000, :]
        Y_meg = Y_meg[:1000, :]
        X_eeg = X_eeg[:1000, :]
        Y_eeg = Y_eeg[:1000, :]
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes[1].imshow(X_meg.T, aspect="auto", cmap=cmap_signal)
    axes[1].set_ylabel("MEG\nchannels", fontsize=fontsize)
    axes[1].set_yticks(np.linspace(0, X_meg.shape[1] - 1, 5, dtype=int))
    axes[1].tick_params(axis="both", labelsize=fontsize - 2)
    axes[1].set_xticks([])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)

    axes[2].imshow(X_eeg.T, aspect="auto", cmap=cmap_signal)
    axes[2].set_ylabel("EEG\nchannels", fontsize=fontsize)
    axes[2].set_yticks(np.linspace(0, X_eeg.shape[1] - 1, 5, dtype=int))
    axes[2].tick_params(axis="both", labelsize=fontsize - 2)
    axes[2].set_xticks([])
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    axes[2].spines["bottom"].set_visible(False)

    time_meg = Y_meg[:, 0]  # continues time
    label_meg = Y_meg[:, 1]  # discrete class
    sc = axes[0].scatter(
        np.arange(Y_meg.shape[0]),
        label_meg,
        c=time_meg,
        s=2,
        cmap=cmap_time,
    )
    ticklabels = [{v: k for k, v in cls_label.items()}[k] for k in np.unique(label_meg)]
    cls_label = {k: v for k, v in cls_label.items() if k in ticklabels}
    axes[0].set_ylabel("Class labels", fontsize=fontsize)
    axes[0].set_xticks([])
    axes[0].set_yticks(
        np.arange(np.unique(label_meg).min(), np.unique(label_meg).max() + 1)
    )
    axes[0].set_yticklabels(ticklabels)
    axes[0].tick_params(axis="both", labelsize=fontsize - 2)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)

    for ax in axes:
        label = ax.yaxis.get_label()
        label.set_rotation_mode("anchor")
        label.set_verticalalignment("top")
        ax.yaxis.set_label_coords(-0.21, 0.5)  # align ylabel position

    cax = axes[0].inset_axes([0.8, 0.98, 0.15, 0.05])  # x, y, width, height
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Time labels (s)", fontsize=fontsize)
    sc.set_clim(0, 2)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["0", "1", "2"])
    cbar.ax.tick_params(labelsize=fontsize - 2)
    return fig, axes


def plot_embedding(
    embedding: np.ndarray,
    y_time: np.ndarray,
    y_class: np.ndarray,
    cls_label: dict[str, int],
    cmap_dict: dict[str, str],
    figsize=(6, 5),
    s: float = 2.0,
    alpha: float = 0.8,
    backend: Literal["mpl", "plotly"] = "mpl",
):
    """
    backend:
        - "mpl"    : 使用 matplotlib 画图，返回 (fig, ax)
        - "plotly" : 使用 plotly 画图，返回 fig（ax 为 None）
    """
    # ====== 通用预处理 ======
    embedding = np.asarray(embedding)
    y_time = np.asarray(y_time).reshape(-1)
    y_class = np.asarray(y_class).reshape(-1)

    inv_cls_label = {v: k for k, v in cls_label.items()}
    unique_classes = np.unique(y_class)

    if embedding.shape[1] < 3:
        raise ValueError(
            f"embedding.shape[1] = {embedding.shape[1]}, 需要至少 3 维才能画 3D"
        )

    # ====== matplotlib backend ======
    if backend == "mpl":
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})

        for c in unique_classes:
            class_name = inv_cls_label[int(c)]
            cmap_name = cmap_dict[class_name]
            cmap = get_cmap(cmap_name)

            mask = y_class == c
            emb_c = embedding[mask]
            t_c = y_time[mask]

            t_min_c, t_max_c = t_c.min(), t_c.max()
            if t_max_c == t_min_c:
                t_norm = np.zeros_like(t_c)
            else:
                t_norm = (t_c - t_min_c) / (t_max_c - t_min_c)

            colors = cmap(t_norm)

            ax.scatter(
                emb_c[:, 0],
                emb_c[:, 1],
                emb_c[:, 2],
                c=colors,
                s=s,
                alpha=alpha,
                edgecolors="none",
            )

        ax.set_xlabel("Latent 1")
        ax.set_ylabel("Latent 2")
        ax.set_zlabel("Latent 3")
        plt.tight_layout()
        return fig, ax

    # ====== plotly backend ======
    elif backend == "plotly":
        fig = go.Figure()

        for c in unique_classes:
            class_name = inv_cls_label[int(c)]
            cmap_name = cmap_dict[class_name]
            cmap = get_cmap(cmap_name)

            mask = y_class == c
            emb_c = embedding[mask]
            t_c = y_time[mask]

            t_min_c, t_max_c = t_c.min(), t_c.max()
            if t_max_c == t_min_c:
                t_norm = np.zeros_like(t_c)
            else:
                t_norm = (t_c - t_min_c) / (t_max_c - t_min_c)

            # 使用 matplotlib colormap 生成 RGBA，然后转为 hex，给 plotly 使用
            rgba = cmap(t_norm)  # (N, 4)
            colors_hex = [mcolors.to_hex(rgba_i) for rgba_i in rgba]

            fig.add_trace(
                go.Scatter3d(
                    x=emb_c[:, 0],
                    y=emb_c[:, 1],
                    z=emb_c[:, 2],
                    mode="markers",
                    marker=dict(
                        size=s,
                        color=colors_hex,  # 直接用每个点的颜色
                        opacity=alpha,
                    ),
                    name=str(class_name),
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="Latent 1",
                yaxis_title="Latent 2",
                zaxis_title="Latent 3",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        return fig

    else:
        raise ValueError(f"Unknown backend: {backend}")


def make_point_colors(
    arr, cmap_by_class, time_range_by_class=None, default_cmap="viridis"
):
    """
    arr: (N,2), arr[:,0]=time(float), arr[:,1]=class(int)
    cmap_by_class: dict, e.g. {0:"Blues", 2:"Reds", ...} or {0: plt.cm.Blues, ...}
    time_range_by_class: dict or None
        - 如果给 dict: {cls: (tmin, tmax)}，就按指定的范围归一化
        - 如果 None: 每个类别用该类别在数据中的 min/max 自动归一化
    return: colors (N,4) RGBA
    """
    t = arr[:, 0].astype(float)
    c = arr[:, 1].astype(int)

    N = len(arr)
    colors = np.zeros((N, 4), dtype=float)

    classes = np.unique(c)
    for cls in classes:
        mask = c == cls
        tt = t[mask]

        # 取该类别的 cmap
        cm = cmap_by_class.get(cls, default_cmap)
        cm = mpl.cm.get_cmap(cm) if isinstance(cm, str) else cm

        # 取归一化范围
        if time_range_by_class is not None and cls in time_range_by_class:
            tmin, tmax = time_range_by_class[cls]
        else:
            tmin, tmax = float(np.min(tt)), float(np.max(tt))

        denom = tmax - tmin
        if denom == 0:
            u = np.zeros_like(tt)
        else:
            u = (tt - tmin) / denom

        u = np.clip(u, 0.0, 1.0)
        colors[mask] = cm(u)

    return colors
