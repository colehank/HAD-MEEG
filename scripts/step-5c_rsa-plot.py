# %%
"""
RSA (Representational Similarity Analysis) Plotting Script

This script visualizes the results of RSA analysis by:
1. Plotting RDMs (Representational Dissimilarity Matrices) for fMRI ROIs and M/EEG data
2. Plotting correlation time courses between M/EEG and fMRI representations
3. Comparing MEG vs EEG representations
"""

# %% Imports
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from joblib import load
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from src.rsa import TimeRDM
from src.utils import get_soi_picks
from src.evo import EvokedSet
from src import DataConfig

# %% Configuration and Constants
cfg = DataConfig()

# Directory setup
SAVE_DIR = cfg.results_root / "rsa"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESOURCE_DIR = Path("./resources")
EVO_DIR = cfg.results_root / "evos" / "grand_evo"
RDMS_DIR = SAVE_DIR / "rdms"
RDMS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
SOIS = ["O", "P", "C", "F", "T", "OT", "all"]  # M/EEG sensors of interest
# ROIS = ["EV", "VS", "DS", "LS"]  # fMRI regions of interest
ROIS = ["EV", "LS"]
ROI_MAP = {"EV": "Early", "VS": "Ventral", "DS": "Dorsal", "LS": "Lateral"}

# Plotting parameters
FONT_SIZE = 12
FONT_PATH = Path("resources") / "Helvetica.ttc"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()

CMAP = ListedColormap(
    [
        (0.556, 0.812, 0.788, 1),  # EV - Early Visual
        (1.000, 0.745, 0.478, 1),  # VS - Ventral Stream
        (0.980, 0.498, 0.435, 1),  # LS - Lateral Stream
        (0.510, 0.690, 0.824, 1),  # DS - Dorsal Stream
    ]
)

RDM_VMIN = -1
RDM_VMAX = 1

# %% Helper Functions


def get_class_feature(evo: EvokedSet, order: np.ndarray, soi: str) -> np.ndarray:
    """
    Extract M/EEG sensor data for specified SOI and class order.

    Parameters
    ----------
    evo : EvokedSet
        The evoked set containing M/EEG data.
    order : np.ndarray
        1D array of class names defining the order.
    soi : str
        Sensor of interest (e.g., 'O', 'P', 'C', 'F', 'T', 'OT').

    Returns
    -------
    np.ndarray
        Feature array with shape (n_classes, n_channels, n_times).
    """
    picks = get_soi_picks(evo[0], soi)
    data = []
    for class_name in order:
        this_evo = evo[evo.metadata["class_name"] == class_name].copy()
        this_data = this_evo.pick(picks).data  # n_channels x n_times
        data.append(this_data)
    return np.array(data)


def get_order(rdm: np.ndarray) -> np.ndarray:
    """
    Compute hierarchical clustering order for RDM visualization.

    Parameters
    ----------
    rdm : np.ndarray
        Representational dissimilarity matrix.

    Returns
    -------
    np.ndarray
        Indices array representing the clustering order.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    rdm = 1 + rdm
    distance_matrix = squareform(rdm, checks=False)
    Z = linkage(distance_matrix, method="average")
    order = leaves_list(Z)
    return order


def plot_rdm(
    rdm: np.ndarray,
    order: np.ndarray | None = None,
    lower_triangle: bool = True,
    normalize: bool = False,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
):
    """
    Plot a representational dissimilarity matrix.

    Parameters
    ----------
    rdm : np.ndarray
        The RDM to plot.
    order : np.ndarray | None
        Optional reordering of matrix indices.
    lower_triangle : bool
        If True, only show lower triangle.
    normalize : bool
        If True, z-score normalize the RDM.
    vmin : float
        Minimum value for colormap.
    vmax : float
        Maximum value for colormap.
    cmap : str
        Colormap name.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
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


def plot_corr(
    conditions: dict,
    time_points: np.ndarray,
    dtype: str = "meg",
    pick: str = "all",
    ax=None,
):
    """
    Plot correlation time courses between M/EEG and fMRI representations.

    Parameters
    ----------
    conditions : dict
        Dictionary containing correlation data for each condition.
    time_points : np.ndarray
        Time points in milliseconds.
    dtype : str
        Data type: 'meg', 'eeg', or 'MEG vs. EEG'.
    pick : str
        Sensor selection: 'all', 'OT', or 'meg-eeg'.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """
    fig = plt.figure(figsize=(6, 2), dpi=300)
    if ax is None:
        ax = fig.add_subplot(111)

    # Y-axis scale configuration for different sensor picks
    pick_scale = {
        "all": {
            "ymin": -0.04,
            "ymax": 0.12,
            "yticks": [0, 0.06, 0.12],
            "loc": None,
        },
        "OT": {
            "ymin": -0.03,
            "ymax": 0.2,
            "yticks": [0, 0.1, 0.2],
            "loc": "upper right",
        },
        "meg-eeg": {
            "ymin": -0.03,
            "ymax": 0.04,
            "yticks": [0, 0.02, 0.04],
            "loc": "upper right",
        },
    }

    # Special handling for MEG vs. EEG comparison
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
            linestyle="-",
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
        print(f"{label} corr_max: {max_corr:.3f} at {max_time:.3f} ms, ci: {ci}")

        # Configure axes
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(time_points[0], time_points[-1])
        ax.set_ylim(pick_scale["meg-eeg"]["ymin"], pick_scale["meg-eeg"]["ymax"])
        ax.set_yticks(pick_scale["meg-eeg"]["yticks"])
        ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE)
        ax.set_ylabel("Spearman r", fontsize=FONT_SIZE)
        ax.axhline(0, color="k", linestyle="--")
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, -0.02),
            fontsize=FONT_SIZE - 3,
            ncol=2,
            frameon=False,
            handlelength=1,
        )
        for legline in ax.get_legend().get_lines():
            legline.set_linewidth(5.0)
        return fig

    # Plot correlations for each ROI
    for label, cond in conditions.items():
        # Skip conditions that don't match the specified data type
        if dtype.upper() not in label:
            continue
        # Skip MEG vs. EEG comparison in ROI plots
        if all(["MEG" in label, "EEG" in label]):
            continue

        ax.plot(
            time_points,
            cond["corrs"],
            color=cond["color"],
            label=label.split(" vs. ")[-1],
            lw=1.5,
            alpha=1,
            linestyle="-",
        )
        ax.fill_between(
            time_points,
            np.array(cond["ci"])[:, 0],
            np.array(cond["ci"])[:, 1],
            color=cond["color"],
            alpha=0.2,
        )

        max_corr = np.max(cond["corrs"])
        max_time = time_points[np.argmax(cond["corrs"])]
        ci = cond["ci"][np.argmax(cond["corrs"])]
        print(f"{label} corr_max: {max_corr:.3f} at {max_time:.3f} ms, ci: {ci}")

    # Configure axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(time_points[0], time_points[-1])

    # Set y-axis limits based on data type and pick
    if dtype == "meg" and pick == "OT":
        ax.set_ylim(pick_scale[pick]["ymin"], pick_scale[pick]["ymax"])
        ax.set_yticks(pick_scale[pick]["yticks"])
    elif dtype == "eeg" and pick == "OT":
        ax.set_ylim(-0.04, 0.10)
        ax.set_yticks([0, 0.10])
    else:
        ax.set_ylim(pick_scale["all"]["ymin"], pick_scale["all"]["ymax"])
        ax.set_yticks(pick_scale["all"]["yticks"])

    ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE)
    ax.set_ylabel("Spearman r", fontsize=FONT_SIZE)
    ax.axhline(0, color="k", linestyle="--", lw=0.8)

    # Configure legend
    if pick == "all":
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, -0.02),
            fontsize=FONT_SIZE - 3,
            ncol=2,
            frameon=False,
            handlelength=1,
        )
    else:
        ax.legend(
            loc=pick_scale[pick]["loc"],
            fontsize=FONT_SIZE - 3,
            frameon=False,
            handlelength=1,
        )

    for legline in ax.get_legend().get_lines():
        legline.set_linewidth(3.0)

    return fig, ax


# %% Load Data
print("Loading data...")
corrs = load(SAVE_DIR / "results.pkl")
evo = load(EVO_DIR / "grand_evo_meg.pkl")
fmri_rdms = np.load(RESOURCE_DIR / "fmri-rdms.npz")

meg_evo = load(EVO_DIR / "grand_evo_meg.pkl")
eeg_evo = load(EVO_DIR / "grand_evo_eeg.pkl")
class_order = fmri_rdms["order"]

# Extract M/EEG features for OT sensors
meg_feat = get_class_feature(
    meg_evo, class_order, soi="OT"
)  # (n_classes, n_channels, n_times)
eeg_feat = get_class_feature(
    eeg_evo, class_order, soi="OT"
)  # (n_classes, n_channels, n_times)
print(f"MEG features shape: {meg_feat.shape}")
print(f"EEG features shape: {eeg_feat.shape}")
# %% Compute M/EEG RDMs
print("Computing M/EEG RDMs...")
# Define time points to show (in sample indices)
show_times = [
    int(i * 200)
    for i in [
        0.3,  # 60 ms (200 ms stimulus + 60 ms = 260 ms)
        0.5,  # 100 ms
        0.7,  # 140 ms
    ]
]

meg_rdms = TimeRDM.compute(
    meg_feat, metric="pearson", normalize=True
)  # (n_times, n_classes, n_classes)
eeg_rdms = TimeRDM.compute(
    eeg_feat, metric="pearson", normalize=True
)  # (n_times, n_classes, n_classes)

meg_show_rdms = meg_rdms[show_times]
eeg_show_rdms = eeg_rdms[show_times]
print(f"MEG RDMs shape: {meg_rdms.shape}")
print(f"EEG RDMs shape: {eeg_rdms.shape}")

# %% Plot and Save fMRI RDMs
print("Plotting fMRI RDMs...")
order = None  # Could use: order = get_order(fmri_rdms['DS'])

for roi in ROIS:
    print(f"  {roi} fMRI RDM shape: {fmri_rdms[roi].shape}")
    fig = plot_rdm(
        fmri_rdms[roi],
        vmin=RDM_VMIN,
        vmax=RDM_VMAX,
        lower_triangle=True,
        order=order,
    )
    fig.savefig(
        RDMS_DIR / f"{roi}.png",
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )

# %% Plot and Save M/EEG RDMs
print("Plotting M/EEG RDMs...")
for idx, t in enumerate(show_times):
    # MEG RDM
    fig = plot_rdm(
        meg_show_rdms[idx],
        vmin=RDM_VMIN,
        vmax=RDM_VMAX,
        lower_triangle=True,
        order=order,
    )
    fig.savefig(
        RDMS_DIR / f"meg_time_{t}.png",
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )

    # EEG RDM
    fig = plot_rdm(
        eeg_show_rdms[idx],
        vmin=RDM_VMIN,
        vmax=RDM_VMAX,
        lower_triangle=True,
        order=order,
    )
    fig.savefig(
        RDMS_DIR / f"eeg_time_{t}.png",
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )

# %% Create and Save Colorbar
print("Creating colorbar...")
norm = Normalize(vmin=RDM_VMIN, vmax=RDM_VMAX)
cmap = plt.get_cmap("RdBu_r")
fig_cbar, ax_cbar = plt.subplots(figsize=(4, 0.4), dpi=300)
fig_cbar.subplots_adjust(bottom=0.6)
fig_cbar.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=ax_cbar,
    orientation="horizontal",
)
ax_cbar.axis("off")
fig_cbar.savefig(
    RDMS_DIR / "colorbar.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
plt.close(fig_cbar)


# %% Prepare Correlation Plotting Data
print("Preparing correlation plotting data...")
pick = "OT"
time_points = evo[0].times * 1e3  # Convert to milliseconds
colors = CMAP.colors

# Build conditions dictionary
conditions = {
    **{
        f"{dtype} vs. {ROI_MAP[roi]}": {
            "corrs": corrs[roi][pick][f"{dtype.lower()}_corr"],
            "ci": corrs[roi][pick][f"{dtype.lower()}_sig"],
            # "color": colors[{"EV": 0, "VS": 1, "LS": 2, "DS": 3}[roi]],
            "color": colors[{"EV": 3, "VS": 2, "LS": 1, "DS": 0}[roi]],
        }
        for roi in ROIS
        for dtype in ("MEG", "EEG")
    },
    "MEG vs. EEG": {
        "corrs": corrs["MEG vs. EEG"]["meg_eeg_corr"],
        "ci": corrs["MEG vs. EEG"]["meg_eeg_sig"],
        "color": "grey",
    },
}
fig, axes = plt.subplots(2, 1, figsize=(6, 4), dpi=300, sharex=True)
plot_corr(conditions, time_points, dtype="meg", pick=pick, ax=axes[0])
plot_corr(conditions, time_points, dtype="eeg", pick=pick, ax=axes[1])
axes[0].spines["bottom"].set_visible(False)
axes[0].get_xaxis().set_visible(False)
for ax in axes:
    ax.get_legend().remove()
plt.show()

fig.savefig(
    SAVE_DIR / f"rsa_meg_eeg_corr_{pick}.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
fig.savefig(
    SAVE_DIR / f"rsa_meg_eeg_corr_{pick}.svg",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
# %%
