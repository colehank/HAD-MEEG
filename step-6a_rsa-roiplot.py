# %%
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from nilearn import plotting
from matplotlib.patches import Patch
from src import DataConfig
# %% Configuration

cfg = DataConfig()
SAVE_DIR = cfg.results_root / "rsa"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ROI_DIR = Path("resources") / "rois"
SURF_DIR = ROI_DIR / "surfaces"

RES_ROI_DIR = SAVE_DIR / "rois"
RES_SOI_DIR = SAVE_DIR / "sois"
RES_ROI_DIR.mkdir(parents=True, exist_ok=True)
RES_SOI_DIR.mkdir(parents=True, exist_ok=True)
bg_color = "lightgray"
EV_NAME = ["V1", "V2", "V3", "V4"]
VS_NAME = [
    "V8",
    "FFC",
    "PIT",
    "VMV1",
    "VMV3",
    "VMV2",
    "VVC",
    "EC",
    "PreS",
    "PeEc",
    "PHA1",
    "PHA3",
    "PHA2",
    "TF",
]
LS_NAME = ["MST", "LO1", "LO2", "MT", "PH", "V4t", "FST", "V3CD", "LO3"]
DS_NAME = [
    "V6",
    "V3A",
    "V7",
    "IPS1",
    "V3B",
    "V6A",
    "7Pm",
    "7Am",
    "7PL",
    "LIPv",
    "VIP",
    "MIP",
    "IP1",
    "IP0",
    "POS2",
    "PCV",
    "POS1",
    "ProS",
    "DVT",
]

ROI_NAMES = {
    "EV": EV_NAME,
    "VS": VS_NAME,
    "DS": DS_NAME,
    "LS": LS_NAME,
}

CMAP = ListedColormap(
    [
        bg_color,  # 0 - non-visual / unclassified
        (0.510, 0.690, 0.824, 1),  # 1 - EV
        # (0.556, 0.812, 0.788, 1),  # 2 - VS
        bg_color,  # 2 - VS
        (1.000, 0.745, 0.478, 1),  # 3 - LS
        # (0.980, 0.498, 0.435, 1),  # 4 - DS
        bg_color,  # 4 - DS
    ]
)
# Utility functions


def to_base_name(label_str: str) -> str:
    """
    Convert label string like 'L_V1_ROI' or 'R_V1_ROI' to base name 'V1'.

    Parameters
    ----------
    label_str : str
        Original label string from roilbl_mmp.csv.

    Returns
    -------
    str
        Base ROI name without hemisphere prefix and '_ROI' suffix.
    """
    s = str(label_str)
    if s.startswith("L_") or s.startswith("R_"):
        s = s[2:]
    if s.endswith("_ROI"):
        s = s[:-4]
    return s


def detect_label_scheme(labels_all: np.ndarray, n_areas: int) -> str:
    """
    Detect how labels are encoded based on the maximum label value.

    Parameters
    ----------
    labels_all : np.ndarray
        1D array of labels for all cortical vertices (concatenated L+R), shape (n_vertices,).
    n_areas : int
        Number of unique areas defined in roilbl_mmp.csv (per hemisphere).

    Returns
    -------
    str
        "PER_HEMI_SHARED" if left/right share 1..n_areas.
        "SPLIT_LR" if left=1..n_areas, right=n_areas+1..2*n_areas.

    Raises
    ------
    ValueError
        If the label encoding does not match expected ranges.
    """
    max_label = int(labels_all.max())

    if max_label <= n_areas:
        scheme = "PER_HEMI_SHARED"  # Left/right share 1..n_areas
    elif max_label <= 2 * n_areas:
        scheme = "SPLIT_LR"  # Left 1..n_areas; right n_areas+1..2*n_areas
    else:
        raise ValueError(
            f"Unexpected label encoding: max_label={max_label}, n_areas={n_areas}"
        )

    return scheme


def build_base_to_rowidx(roi_all_names: pd.DataFrame) -> dict[str, int]:
    """
    Build a mapping from base ROI name (e.g. 'V1') to row index in roilbl_mmp.csv.

    Parameters
    ----------
    roi_all_names : pandas.DataFrame
        DataFrame loaded from roilbl_mmp.csv (two columns: left/right).

    Returns
    -------
    dict[str, int]
        Mapping from base name to row index (0-based).
    """
    col_L, col_R = roi_all_names.columns
    base_to_rowidx: dict[str, int] = {}

    for idx, row in roi_all_names.iterrows():
        for col in (col_L, col_R):
            base = to_base_name(row[col])
            base_to_rowidx[base] = idx

    return base_to_rowidx


def make_label_id_getter(
    base_to_rowidx: dict[str, int], label_scheme: str, n_areas: int
):
    """
    Create a function that maps a base ROI name to label IDs, given a label scheme.

    Parameters
    ----------
    base_to_rowidx : dict[str, int]
        Base name -> row index mapping.
    label_scheme : str
        Label scheme, e.g. "PER_HEMI_SHARED" or "SPLIT_LR".
    n_areas : int
        Number of areas per hemisphere.

    Returns
    -------
    Callable[[str], list[int]]
        Function that returns label indices for a given base ROI name.
    """

    def get_label_ids_for_basename(base: str) -> list[int]:
        """
        Return label IDs corresponding to a given base name (e.g. 'V1').

        Depending on the scheme, this may return one ID (shared across hemispheres)
        or two IDs (left/right separately).
        """
        if base not in base_to_rowidx:
            return []

        row_idx = base_to_rowidx[base]  # 0-based
        if label_scheme == "PER_HEMI_SHARED":
            # Left/right share 1..n_areas (brain structure distinguishes hemispheres)
            return [row_idx + 1]
        elif label_scheme == "SPLIT_LR":
            # Left: 1..n_areas; Right: n_areas+1..2*n_areas
            return [row_idx + 1, row_idx + 1 + n_areas]
        else:
            return []

    return get_label_ids_for_basename


def build_system_map(
    labels_hemi_full: np.ndarray,
    roi_names: dict[str, list[str]],
    get_label_ids_for_basename,
) -> np.ndarray:
    """
    Build per-vertex system map for a single hemisphere.

    Parameters
    ----------
    labels_hemi_full : np.ndarray
        Label array on the full 32k surface for one hemisphere, shape (n_vertices,).
        Non-cortical vertices should be 0.
    roi_names : dict[str, list[str]]
        Dictionary with keys 'EV', 'VS', 'LS', 'DS', each mapped to a list of base ROI names.
    get_label_ids_for_basename : Callable[[str], list[int]]
        Function that returns label IDs for a given base ROI name.

    Returns
    -------
    np.ndarray
        Integer array of system labels, shape (n_vertices,):
            0 = non-visual / unclassified
            1 = EV
            2 = VS
            3 = LS
            4 = DS
    """
    sys_map = np.zeros_like(labels_hemi_full, dtype=int)

    def mark_group(group_names: list[str], group_id: int) -> None:
        all_ids: list[int] = []
        for base in group_names:
            ids = get_label_ids_for_basename(base)
            if not ids:
                continue
            all_ids.extend(ids)

        if not all_ids:
            return

        all_ids_arr = np.array(all_ids, dtype=int)
        mask = np.isin(labels_hemi_full, all_ids_arr)
        sys_map[mask] = group_id

    mark_group(roi_names["EV"], 1)
    mark_group(roi_names["VS"], 2)
    mark_group(roi_names["LS"], 3)
    mark_group(roi_names["DS"], 4)

    return sys_map


def plot_systems_on_surface(
    coords_l: np.ndarray,
    faces_l: np.ndarray,
    coords_r: np.ndarray,
    faces_r: np.ndarray,
    sys_l: np.ndarray,
    sys_r: np.ndarray,
    bg: np.ndarray,
    view: str = "dorsal",
) -> None:
    """
    Plot visual system classification on left and right cortical surfaces.

    Parameters
    ----------
    coords_l, faces_l : np.ndarray
        Left hemisphere surface coordinates and faces.
    coords_r, faces_r : np.ndarray
        Right hemisphere surface coordinates and faces.
    sys_l, sys_r : np.ndarray
        Per-vertex system labels (0..4) for left/right hemispheres.
    bg : np.ndarray
        Background map (e.g., sulcal depth) for shading.
        NOTE: For consistency with the original script, the same `bg` is used
        for both hemispheres.
    """

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 6))

    plotting.plot_surf(
        surf_mesh=(coords_l, faces_l),
        surf_map=sys_l,
        cmap=CMAP,
        vmin=0,
        vmax=4,
        darkness=None,
        bg_map=bg,
        bg_on_data=True,
        avg_method="median",
        axes=axes[0],
        colorbar=False,
        hemi="left",
        view=view,
    )

    plotting.plot_surf(
        surf_mesh=(coords_r, faces_r),
        surf_map=sys_r,
        cmap=CMAP,
        vmin=0,
        vmax=4,
        darkness=None,
        bg_map=bg,
        bg_on_data=True,
        avg_method="median",
        axes=axes[1],
        colorbar=False,
        hemi="right",
        view=view,
    )

    plt.tight_layout()
    plt.close(fig)
    return fig


# Main workflow

# %%
if __name__ == "__main__":
    # 1. Load surfaces

    surf_l = nib.load(str(SURF_DIR / "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"))
    surf_r = nib.load(str(SURF_DIR / "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"))

    coords_l = surf_l.darrays[0].data
    faces_l = surf_l.darrays[1].data
    coords_r = surf_r.darrays[0].data
    faces_r = surf_r.darrays[1].data

    nL = coords_l.shape[0]
    nR = coords_r.shape[0]

    # 2. Load Glasser labels (59412 cortical vertices)

    mat = sio.loadmat(ROI_DIR / "MMP_mpmLR32k.mat")
    labels_all = np.squeeze(mat["glasser_MMP"]).astype(int)  # (59412,)

    # 3. Use template.dtseries.nii to find cortical vertex indices in 32k surface

    temp_cifti = nib.load(str(ROI_DIR / "template.dtseries.nii"))
    hdr = temp_cifti.header

    brain_models = [bm for bm in hdr.matrix[1].brain_models]

    # Find left/right cortex brain models
    bm_L = [bm for bm in brain_models if "CORTEX_LEFT" in bm.brain_structure][0]
    bm_R = [bm for bm in brain_models if "CORTEX_RIGHT" in bm.brain_structure][0]

    # Indices into the 32k surface (0..32491), usually ~29k for cortex
    vert_idx_L = np.asarray(bm_L.vertex_indices)
    vert_idx_R = np.asarray(bm_R.vertex_indices)

    nL_cortex = vert_idx_L.size
    nR_cortex = vert_idx_R.size

    assert nL_cortex + nR_cortex == labels_all.shape[0], (
        "Number of cortical CIFTI vertices does not match label array length."
    )

    # 4. Build full 32k label arrays (non-cortical vertices = 0)

    labels_l_full = np.zeros(nL, dtype=int)
    labels_r_full = np.zeros(nR, dtype=int)

    labels_l_full[vert_idx_L] = labels_all[:nL_cortex]
    labels_r_full[vert_idx_R] = labels_all[nL_cortex:]

    # 5. Load roilbl_mmp.csv and build base-name -> label-id mapping

    roi_all_names = pd.read_csv(ROI_DIR / "roilbl_mmp.csv")

    n_areas = roi_all_names.shape[0]  # typically 180

    base_to_rowidx = build_base_to_rowidx(roi_all_names)
    label_scheme = detect_label_scheme(labels_all, n_areas)
    get_label_ids_for_basename = make_label_id_getter(
        base_to_rowidx=base_to_rowidx,
        label_scheme=label_scheme,
        n_areas=n_areas,
    )

    # 6. Build per-vertex system maps

    sys_l = build_system_map(
        labels_hemi_full=labels_l_full,
        roi_names=ROI_NAMES,
        get_label_ids_for_basename=get_label_ids_for_basename,
    )
    sys_r = build_system_map(
        labels_hemi_full=labels_r_full,
        roi_names=ROI_NAMES,
        get_label_ids_for_basename=get_label_ids_for_basename,
    )
    # %%
    # 7. Plot

    bg = nib.load(str(SURF_DIR / "996782.L.sulc.32k_fs_LR.shape.gii")).darrays[0].data

    figs = {}
    views = ["anterior", "posterior", "medial", "lateral", "dorsal", "ventral"]
    for view in views:
        figs[view] = plot_systems_on_surface(
            coords_l=coords_l,
            faces_l=faces_l,
            coords_r=coords_r,
            faces_r=faces_r,
            sys_l=sys_l,
            sys_r=sys_r,
            bg=bg,
            view=view,
        )
    # %%
    flat_surf_l = nib.load(str(SURF_DIR / "S1200.L.flat.32k_fs_LR.surf.gii"))
    coords_l_flat = flat_surf_l.darrays[0].data
    faces_l_flat = flat_surf_l.darrays[1].data
    coords_r_flat = (
        nib.load(str(SURF_DIR / "S1200.R.flat.32k_fs_LR.surf.gii")).darrays[0].data
    )
    faces_r_flat = (
        nib.load(str(SURF_DIR / "S1200.R.flat.32k_fs_LR.surf.gii")).darrays[1].data
    )
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 6))
    plotting.plot_surf(
        surf_mesh=(coords_l_flat, faces_l_flat),
        surf_map=sys_l,
        cmap=CMAP,
        vmin=0,
        vmax=4,
        darkness=None,
        bg_map=bg,
        bg_on_data=True,
        avg_method="median",
        hemi="left",
        # view="dorsal",
        axes=axes[0],
        colorbar=False,
    )
    plotting.plot_surf(
        surf_mesh=(coords_r_flat, faces_r_flat),
        surf_map=sys_r,
        cmap=CMAP,
        vmin=0,
        vmax=4,
        darkness=None,
        bg_map=bg,
        bg_on_data=True,
        avg_method="median",
        hemi="right",
        # view="dorsal",
        axes=axes[1],
        colorbar=False,
    )
    # handles = [
    #     # Patch(color="ghostwhite", label="Non-visual / Unclassified"),
    #     Patch(color=(0.556, 0.812, 0.788, 1), label='Early'),
    #     Patch(color=(1.000, 0.745, 0.478, 1), label='Ventral'),
    #     Patch(color=(0.510, 0.690, 0.824, 1), label='Dorsal'),
    #     Patch(color=(0.980, 0.498, 0.435, 1), label='Lateral'),
    # ]
    # fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False)
    axes[0].view_init(elev=90, azim=270)  # 俯视 flat 平面，水平旋转180°
    axes[1].view_init(elev=90, azim=270)  # 俯视 flat 平面，水平旋转180°
    fig.subplots_adjust(wspace=0)
    plt.tight_layout()
    figs["flat"] = fig
    # %%
    for view, fig in figs.items():
        fig.savefig(
            RES_ROI_DIR / f"visual_{view}.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            RES_ROI_DIR / f"visual_{view}.svg",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    # %%
    fig_legend = plt.figure(figsize=(6, 1), dpi=300)
    handles = [
        # Patch(color="ghostwhite", label="Non-visual / Unclassified"),
        Patch(color=(0.556, 0.812, 0.788, 1), label="Early"),
        Patch(color=(1.000, 0.745, 0.478, 1), label="Ventral"),
        Patch(color=(0.510, 0.690, 0.824, 1), label="Dorsal"),
        Patch(color=(0.980, 0.498, 0.435, 1), label="Lateral"),
    ]
    fig_legend.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=12)
    plt.tight_layout()
    fig_legend.savefig(
        RES_ROI_DIR / "legend.png", dpi=300, bbox_inches="tight", transparent=True
    )
