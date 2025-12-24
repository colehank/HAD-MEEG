from .stat import sliding_1sample_ttest
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def plot_sliding_acc(
    np_Mscore,
    np_Escore,
    times,
    fontsize,
    title: str = "Decoding Accuracy (%)",
    vmin=40,
    vmax=70,
    show_individual_accuracy=True,
    color_meg: str | tuple = (0.121, 0.466, 0.705),
    color_eeg: str | tuple = (1.000, 0.498, 0.054),
    verbose: bool = True,
) -> plt.Figure:
    times = times * 1000
    mean_Mscores = np.mean(np_Mscore, axis=0) * 100
    mean_Escores = np.mean(np_Escore, axis=0) * 100
    Msig_times = times[sliding_1sample_ttest(np_Mscore, np_Mscore.shape[1])[2]]
    Esig_times = times[sliding_1sample_ttest(np_Escore, np_Escore.shape[1])[2]]

    if len(Msig_times) == 0 or len(Esig_times) == 0:
        Msig_times = [0]
        Esig_times = [0]
    Minitil_sig_t = Msig_times[0]
    Einitil_sig_t = Esig_times[0]
    Mmax_performance_t = times[np.argmax(mean_Mscores)]
    Emax_performance_t = times[np.argmax(mean_Escores)]
    if verbose:
        print(
            f"MEG has {len(Msig_times)} sig times, EEG has {len(Esig_times)} sig times"
        )
        print(
            f"MEG intial_sig_time:{Minitil_sig_t},max_performance{np.max(mean_Mscores)} at {Mmax_performance_t}ms"
        )
        print(
            f"EEG intial_sig_time:{Einitil_sig_t},max_performance{np.max(mean_Escores)} at {Emax_performance_t}ms"
        )

    min_y = min(np.min(mean_Mscores), np.min(mean_Escores))
    max_y = max(np.max(mean_Mscores), np.max(mean_Escores))
    scatter_space = (max_y - 47) / 12
    yticks = np.arange(np.floor(min_y / 5) * 5, np.ceil(max_y / 5) * 5 + 1, 5)

    scatter_y = np.linspace(50 - scatter_space, 50, 5)
    meg_y = scatter_y[2]
    eeg_y = scatter_y[0]

    plt.close("all")

    if show_individual_accuracy:
        gs = GridSpec(6, 6)
        fig = plt.figure(figsize=(7, 3))
        ax1 = fig.add_subplot(gs[:, :4])
        ax2 = fig.add_subplot(gs[:3, 4:])
        ax3 = fig.add_subplot(gs[3:, 4:])

        # ax1.set_yticks([50, 55, 60])
        ax1.set_xlim([-100, 2000])
        ax1.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])

        ax1.axhline(50, color="k", linestyle="--")
        ax1.plot(times, mean_Mscores, label="MEG", color=color_meg, lw=2)
        ax1.plot(times, mean_Escores, label="EEG", color=color_eeg, lw=2)
        ax1.set_yticks(yticks)

        ax1.scatter(
            Msig_times,
            [meg_y] * len(Msig_times),
            color=color_meg,
            marker="o",
            s=10,
            alpha=0.35,
        )  # 修改这里
        ax1.scatter(
            Esig_times,
            [eeg_y] * len(Esig_times),
            color=color_eeg,
            marker="o",
            s=10,
            alpha=0.35,
        )  # 修改这里

        ax1.set_xlabel("Time (ms)", fontsize=fontsize)
        ax1.set_ylabel(title, fontsize=fontsize)
        ax1.legend(loc="upper left", fontsize=fontsize)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        for i, (decoaccuracy, ax, title) in enumerate(
            zip([np_Mscore, np_Escore], [ax2, ax3], ["MEG", "EEG"])
        ):
            decoaccuracy = decoaccuracy * 100
            im = ax.imshow(
                decoaccuracy,
                aspect="auto",
                cmap="coolwarm",
                extent=[times[0], times[-1], 0.5, decoaccuracy.shape[0] + 0.5],
                interpolation="nearest",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            # ax.set_yticks([])
            # ax.set_yticklabels([])
            if title == "EEG":
                ax.set_xticks([0, 500, 1000, 1500, 2000])
                ax.set_xticklabels([0, 500, 1000, 1500, 2000], fontsize=fontsize - 2)
                ax.set_xlabel("Time (ms)", fontsize=fontsize)
                eeg_subs = [
                    1,
                    2,
                    3,
                    4,
                    6,
                    7,
                    8,
                    9,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    29,
                    30,
                    31,
                ]
                ax.set_yticks(np.arange(1, len(eeg_subs) + 1, 3))
                ax.set_yticklabels(
                    eeg_subs[::3], fontsize=fontsize - 5.5, rotation=0, ha="right"
                )
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel("")
                ax.set_yticks(np.arange(1, decoaccuracy.shape[0] + 1, 3))
                ax.set_yticklabels(
                    np.arange(1, decoaccuracy.shape[0] + 1, 3),
                    fontsize=fontsize - 5.5,
                    rotation=0,
                    ha="right",
                )
            ax.set_ylabel(f"{title} Subject", fontsize=fontsize)
        cbar_ax = fig.add_axes(
            [1, 0.2, 0.01, 0.75]
        )  # Adjust the position and size of the colorbar
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Decoding Accuracy (%)", fontsize=fontsize)

    else:
        fig, ax1 = plt.subplots(dpi=600, figsize=(7, 3))
        ax1.set_xlim([-100, 2000])
        ax1.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])

        ax1.axhline(50, color="k", linestyle="--")
        ax1.plot(times, mean_Mscores, label="MEG", color=color_meg, lw=2)
        ax1.plot(times, mean_Escores, label="EEG", color=color_eeg, lw=2)
        ax1.set_yticks(yticks)

        ax1.scatter(
            Msig_times,
            [meg_y] * len(Msig_times),
            color=color_meg,
            marker="o",
            s=10,
            alpha=0.35,
        )
        ax1.scatter(
            Esig_times,
            [eeg_y] * len(Esig_times),
            color=color_eeg,
            marker="o",
            s=10,
            alpha=0.35,
        )

        ax1.set_xlabel("Time (ms)", fontsize=fontsize)
        ax1.set_ylabel(title, fontsize=fontsize)
        ax1.legend(loc="upper left", fontsize=fontsize)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(loc="upper left", fontsize=fontsize, framealpha=0.5)

    # plt.tight_layout()
    return fig
