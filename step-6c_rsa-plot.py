# %%
import numpy as np
from pathlib import Path
from joblib import load
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from src.rsa import TimeRDM
from src.utils import get_soi_picks
from src.evo import EvokedSet

# %%
RESOURCE_DIR = Path("./resources")
EVO_DIR = Path("../HAD-MEEG_results/grand_evo")
RES_DIR = Path("../HAD-MEEG_results/rsa")
SOIS = ["O", "P", "C", "F", "T", "OT", "all"]  # (M/EEG) sensor of interests
ROIS = ["EV", "VS", "DS", "LS"]  # (fMRI) regions of interest
ROI_MAP = {"EV": "Early", "VS": "Ventral", "DS": "Dorsal", "LS": "Lateral"}

FONT_SIZE = 12


# %%
def get_class_feature(
    evo: EvokedSet,
    order: np.ndarray,
    soi: str,
):
    """Get the M/EEG's sensor data of
    the specified SOI and class order.

    Parameters
    ----------
    evo : EvokedSet
        The evoked set.
    order : np.ndarray
        The order of the classes in 1d array,
        each element is the class_name(str).
    soi : str
        The SOI name,
        e.g., 'O', 'P', 'C', 'F', 'T', 'OT'.
    """
    picks = get_soi_picks(evo[0], soi)
    data = []
    for ori in order:
        this_evo = evo[evo.metadata["class_name"] == ori].copy()
        this = this_evo.pick(picks).data  # nchan x ntime
        data.append(this)
    return np.array(data)  # nclass x nchan x ntime


def plot_rdm(
    rdm: np.ndarray,
    order: np.ndarray | None = None,
    lower_triangle: bool = True,
    normalize: bool = False,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
):
    rdm = rdm.astype(float)
    if order is not None:
        rdm = rdm[np.ix_(order, order)]
    if normalize:
        finite_mask = np.isfinite(rdm)
        if finite_mask.any():
            mean = np.mean(rdm[finite_mask])
            std = np.std(rdm[finite_mask])
            if not np.isfinite(std) or std == 0:
                std = 1.0
            normalized = np.zeros_like(rdm, dtype=float)
            normalized[finite_mask] = (rdm[finite_mask] - mean) / std
            rdm = normalized
        else:
            rdm = np.zeros_like(rdm, dtype=float)

    if lower_triangle:
        mask = np.triu(np.ones_like(rdm, dtype=bool), k=0)
        rdm = np.ma.array(rdm, mask=mask)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.matshow(rdm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.close(fig)
    return fig


def get_order(rdm: np.ndarray):
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    rdm = 1 + rdm
    distance_matrix = squareform(rdm, checks=False)
    Z = linkage(distance_matrix, method="average")
    order = leaves_list(Z)
    return order


# %%
corrs = load(RES_DIR / "results.pkl")
evo = load(EVO_DIR / "grand_evo_meg.pkl")
fmri_rdms = np.load(RESOURCE_DIR / "fmri-rdms.npz")
fmri_rdms_to = RES_DIR / "fmri_rdms"
fmri_rdms_to.mkdir(parents=True, exist_ok=True)

meg_evo = load(EVO_DIR / "grand_evo_meg.pkl")
eeg_evo = load(EVO_DIR / "grand_evo_eeg.pkl")
class_order = fmri_rdms["order"]
meg_feat = get_class_feature(meg_evo, class_order, soi="all")  # nclass x nchan x ntime
eeg_feat = get_class_feature(eeg_evo, class_order, soi="all")  # nclass x nchan x ntime
# %%
show_times = [
    int(i * 200)
    for i in [
        0.3,  # 200 ms after stimulus onset
        0.5,  # 400 ms after stimulus onset
        0.7,  # 600 ms after stimulus onset
    ]
]

meg_rdms = TimeRDM.compute(
    meg_feat, metric="pearson", normalize=True
)  # ntime x nclass x nclass
eeg_rdms = TimeRDM.compute(
    eeg_feat, metric="pearson", normalize=True
)  # ntime x nclass x nclass

meg_show_rdms = meg_rdms[show_times]
eeg_show_rdms = eeg_rdms[show_times]
# %%
vmin = -1
vmax = 1
# order = get_order(fmri_rdms['DS'])
order = None
for roi in ROIS:
    print(f"{roi} fmri_rdm shape: {fmri_rdms[roi].shape}")
    fig = plot_rdm(
        fmri_rdms[roi], vmin=vmin, vmax=vmax, lower_triangle=True, order=order
    )
    fig.savefig(
        fmri_rdms_to / f"{roi}.png", bbox_inches="tight", transparent=True, dpi=300
    )
for idx, t in enumerate(show_times):
    fig = plot_rdm(
        meg_show_rdms[idx], vmin=vmin, vmax=vmax, lower_triangle=True, order=order
    )
    fig.savefig(
        fmri_rdms_to / f"meg_time_{t}.png",
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )
    fig = plot_rdm(
        eeg_show_rdms[idx], vmin=vmin, vmax=vmax, lower_triangle=True, order=order
    )
    fig.savefig(
        fmri_rdms_to / f"eeg_time_{t}.png",
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )
# %%
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")
fig_cbar, ax_cbar = plt.subplots(figsize=(4, 0.4), dpi=300)
fig_cbar.subplots_adjust(bottom=0.6)
fig_cbar.colorbar(
    ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar, orientation="horizontal"
)
ax_cbar.axis("off")
fig_cbar.savefig(
    fmri_rdms_to / "colorbar.png", bbox_inches="tight", transparent=True, dpi=300
)
plt.close(fig_cbar)
# %%
pick = "OT"
time_points = evo[0].times * 1e3  # in ms
CMAP = ListedColormap(
    [
        (0.556, 0.812, 0.788, 1),  # 1 - EV
        (1.000, 0.745, 0.478, 1),  # 2 - VS
        (0.980, 0.498, 0.435, 1),  # 3 - LS
        (0.510, 0.690, 0.824, 1),  # 4 - DS
    ]
)
colors = CMAP.colors
conditions = {
    f"MEG vs. {ROI_MAP['EV']}": {
        "corrs": corrs["EV"][pick]["meg_corr"],
        "ci": corrs["EV"][pick]["meg_sig"],
        "color": colors[0],
    },
    f"MEG vs. {ROI_MAP['VS']}": {
        "corrs": corrs["VS"][pick]["meg_corr"],
        "ci": corrs["VS"][pick]["meg_sig"],
        "color": colors[1],
    },
    f"MEG vs. {ROI_MAP['DS']}": {
        "corrs": corrs["DS"][pick]["meg_corr"],
        "ci": corrs["DS"][pick]["meg_sig"],
        "color": colors[3],
    },
    f"MEG vs. {ROI_MAP['LS']}": {
        "corrs": corrs["LS"][pick]["meg_corr"],
        "ci": corrs["LS"][pick]["meg_sig"],
        "color": colors[2],
    },
    f"EEG vs. {ROI_MAP['EV']}": {
        "corrs": corrs["EV"][pick]["eeg_corr"],
        "ci": corrs["EV"][pick]["eeg_sig"],
        "color": colors[0],
    },
    f"EEG vs. {ROI_MAP['VS']}": {
        "corrs": corrs["VS"][pick]["eeg_corr"],
        "ci": corrs["VS"][pick]["eeg_sig"],
        "color": colors[1],
    },
    f"EEG vs. {ROI_MAP['DS']}": {
        "corrs": corrs["DS"][pick]["eeg_corr"],
        "ci": corrs["DS"][pick]["eeg_sig"],
        "color": colors[3],
    },
    f"EEG vs. {ROI_MAP['LS']}": {
        "corrs": corrs["LS"][pick]["eeg_corr"],
        "ci": corrs["LS"][pick]["eeg_sig"],
        "color": colors[2],
    },
    "MEG vs. EEG": {
        "corrs": corrs["MEG vs. EEG"]["meg_eeg_corr"],
        "ci": corrs["MEG vs. EEG"]["meg_eeg_sig"],
        "color": "grey",
    },
}


def plot_corr(
    conditions,
    time_points,
    dtype="meg",
    pick="all",
):
    plt.close("all")
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = fig.add_subplot(111)
    linestyle = "-"
    pick_scale = {
        "all": {
            "ymin": -0.04,
            "ymax": 0.15,
            "yticks": [0, 0.05, 0.10, 0.15],
            "loc": None,
        },
        "OT": {
            "ymin": -0.03,
            "ymax": 0.21,
            "yticks": [0, 0.11, 0.22],
            "loc": "upper right",
        },
        "meg-eeg": {
            "ymin": -0.03,
            "ymax": 0.04,
            "yticks": [0, 0.02, 0.04],
            "loc": "upper right",
        },
    }

    if dtype == "MEG vs. EEG":
        cond = conditions[dtype]
        label = dtype
        ax.plot(
            time_points,
            cond["corrs"],
            color=cond["color"],
            label=label,
            lw=1,
            alpha=1,
            linestyle=linestyle,
        )
        ax.fill_between(
            time_points,
            np.array(cond["ci"])[:, 0],
            np.array(cond["ci"])[:, 1],
            color=cond["color"],
            alpha=0.1,
        )
        max_corr = np.max(cond["corrs"])
        max_time = time_points[np.argmax(cond["corrs"])]
        ci = cond["ci"][np.argmax(cond["corrs"])]
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(time_points[0], time_points[-1])
        ax.set_ylim(pick_scale["meg-eeg"]["ymin"], pick_scale["meg-eeg"]["ymax"])
        ax.set_yticks(pick_scale["meg-eeg"]["yticks"])
        # ax.set_yticklabels([0, 0.05, 0.10, 0.15], fontsize=FONT_SIZE)
        ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE)
        ax.set_ylabel("Spearman r", fontsize=FONT_SIZE)
        ax.axhline(0, color="k", linestyle="--")
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, -0.02),
            fontsize=FONT_SIZE - 3,
            ncol=2,
            frameon=False,
            # prop={'weight': 'bold'},
            handlelength=1,
            # handleheight=2
        )
        for legline in ax.get_legend().get_lines():
            legline.set_linewidth(5.0)
        print(f"{label} corr_max: {max_corr:.3f} at {max_time:3f} ms, ci: {ci}")
        return fig
    for label, cond in conditions.items():
        # linestyle = '-' if 'MEG' in label else '--'
        if dtype.upper() not in label:
            continue
        # alpha = 0.5 if label == 'MEG vs. EEG' else 1
        if all(["MEG" in label, "EEG" in label]):
            continue
        ax.plot(
            time_points,
            cond["corrs"],
            color=cond["color"],
            label=label,
            lw=1,
            alpha=1,
            linestyle=linestyle,
        )
        ax.fill_between(
            time_points,
            np.array(cond["ci"])[:, 0],
            np.array(cond["ci"])[:, 1],
            color=cond["color"],
            alpha=0.1,
        )

        max_corr = np.max(cond["corrs"])
        max_time = time_points[np.argmax(cond["corrs"])]
        ci = cond["ci"][np.argmax(cond["corrs"])]
        print(f"{label} corr_max: {max_corr:.3f} at {max_time:3f} ms, ci: {ci}")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(time_points[0], time_points[-1])
    if dtype == "meg":
        ax.set_ylim(pick_scale[pick]["ymin"], pick_scale[pick]["ymax"])
        ax.set_yticks(pick_scale[pick]["yticks"])
    else:
        ax.set_ylim(-0.04, 0.10)
        ax.set_yticks([0, 0.05, 0.10])
    ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE)
    ax.set_ylabel("Spearman r", fontsize=FONT_SIZE)
    ax.axhline(0, color="k", linestyle="--")
    if pick == "all":
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, -0.02),
            fontsize=FONT_SIZE - 3,
            ncol=2,
            frameon=False,
            # prop={'weight': 'bold'},
            handlelength=1,
            # handleheight=2
        )
    else:
        ax.legend(
            loc=pick_scale[pick]["loc"],
            fontsize=FONT_SIZE - 3,
            frameon=False,
            # prop={'weight': 'bold'},
            handlelength=1,
            # handleheight=2
        )
    for legline in ax.get_legend().get_lines():
        legline.set_linewidth(3.0)
    return fig


# %%
eeg_fig = plot_corr(conditions, time_points, dtype="eeg", pick=pick)
eeg_fig.savefig(
    RES_DIR / f"rsa_eeg_{pick}_plot.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
# %%
meg_fig = plot_corr(conditions, time_points, dtype="meg", pick=pick)
meg_fig.savefig(
    RES_DIR / f"rsa_meg_{pick}_plot.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
# %%
meg_eeg_fig = plot_corr(conditions, time_points, dtype="MEG vs. EEG", pick=pick)
meg_eeg_fig.savefig(
    RES_DIR / f"rsa_meg_eeg_{pick}_plot.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)

# %%
