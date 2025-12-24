# %%
from src.config import DataConfig
import pandas as pd
import mne
import numpy as np
from scipy.stats import norm

from src.decoding import EpochsDecoder, plot_sliding_acc
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
import pickle as pkl

RANDOM_STATE = 42
N_JOBS = 8
cfg = DataConfig()
df = pd.read_csv(cfg.bids_root / "stimuli" / "dim_scores_binarized.csv")
SAVE_DIR = cfg.results_root / "decoding"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Plot configuration constants
FONT_SIZE: int = 12
FONT_PATH = Path("resources/Helvetica.ttc")
fm.fontManager.addfont(str(FONT_PATH))
plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()

COLORMAP = "Spectral"
cmap = plt.get_cmap(COLORMAP, 10)
colors = [cmap(i / (10 - 1)) for i in range(10)]
COLOR_MEG = colors[2]
COLOR_EEG = colors[-3]

# Decoding accuracy plot range
PLOT_VMIN = 40
PLOT_VMAX = 70


# %%
def load_epo(sub: str, modality: str) -> mne.Epochs:
    """
    Load epochs data for a given subject and modality.

    Args:
        sub: Subject ID
        modality: Data modality ('eeg' or 'meg')

    Returns:
        Loaded MNE Epochs object

    Raises:
        ValueError: If no data found for the subject/modality combination
        FileNotFoundError: If the epochs file doesn't exist
    """
    try:
        epo_path = cfg.source_df.query("subject == @sub and session == @modality").iloc[
            0
        ]["epochs"]
    except IndexError:
        raise ValueError(
            f"No data found for subject '{sub}' with modality '{modality}'"
        )

    try:
        epochs = mne.read_epochs(epo_path, preload=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Epochs file not found: {epo_path}")

    return epochs


# def split_cls_by_dim(dim: str, thre:float=0.5):
#     upper = df[df[dim]>=thre]['class'].tolist()
#     lower = df[df[dim]<thre]['class'].tolist()
#     return upper, lower


def split_cls_by_dim(
    dim: str,
    thre: float = 0.2,
) -> tuple[list[str], list[str], dict]:
    """
    Split classes into high and low groups based on a dimension using z-score thresholding.

    Args:
        dim: Dimension name (column in the dataframe)
        thre: Percentile threshold for splitting (default 0.2 = top/bottom 20%)

    Returns:
        Tuple of (high_classes, low_classes, stats_dict) where:
        - high_classes: List of class names with high scores
        - low_classes: List of class names with low scores
        - stats_dict: Dictionary containing 'mu', 'sigma', and 'z_thre'
    """
    t = norm.ppf(1 - thre)
    s = df[dim].astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    z = (s - mu) / sigma

    high = df[z >= t]["class"].tolist()
    low = df[z < -t]["class"].tolist()
    return high, low, {"mu": mu, "sigma": sigma, "z_thre": float(t)}


def split_epochs_by_dim(
    epochs: mne.Epochs,
    dim: str,
    thre: float = 0.5,
    balance_epo: bool = True,
) -> tuple[mne.Epochs, mne.Epochs]:
    """
    Split epochs into upper and lower groups based on a dimension.

    Args:
        epochs: MNE Epochs object with metadata
        dim: Dimension name for splitting
        thre: Split threshold (default 0.5)
        balance_epo: Whether to balance epoch counts between groups

    Returns:
        Tuple of (upper_epochs, lower_epochs)
    """
    upper, lower, _ = split_cls_by_dim(dim, thre)
    meta = epochs.metadata

    epo_upper = epochs[meta["class_name"].isin(upper)]
    epo_lower = epochs[meta["class_name"].isin(lower)]

    if balance_epo:
        np.random.seed(RANDOM_STATE)

        n_min = min(len(epo_upper), len(epo_lower))

        idx_up = np.random.choice(len(epo_upper), size=n_min, replace=False)
        idx_low = np.random.choice(len(epo_lower), size=n_min, replace=False)

        epo_upper = epo_upper[idx_up]
        epo_lower = epo_lower[idx_low]

    return epo_upper, epo_lower


def split_epochs_by_class(
    epochs: mne.Epochs,
    class_col: str,
    class_name: str,
    balance_epo: bool = True,
) -> tuple[mne.Epochs, mne.Epochs]:
    """
    Split epochs into class vs non-class groups.

    Args:
        epochs: MNE Epochs object with metadata
        class_col: Column name for class classification
        class_name: Name of the target class
        balance_epo: Whether to balance epoch counts between groups

    Returns:
        Tuple of (class_epochs, non_class_epochs)
    """
    meta = epochs.metadata

    epo_class = epochs[meta[class_col] == class_name]
    epo_nonclass = epochs[meta[class_col] != class_name]
    if balance_epo:
        np.random.seed(RANDOM_STATE)

        n_min = min(len(epo_class), len(epo_nonclass))

        idx_class = np.random.choice(len(epo_class), size=n_min, replace=False)
        idx_nonclass = np.random.choice(len(epo_nonclass), size=n_min, replace=False)

        epo_class = epo_class[idx_class]
        epo_nonclass = epo_nonclass[idx_nonclass]

    return epo_class, epo_nonclass


def _decode_one_sub(
    sub: str,
    source_df: pd.DataFrame,
    dim: str,
    modality: str,
    pick_type: str,
    cv: int,
    inner_n_jobs: int = 1,
    pca: bool = True,
    split_thre: float = 0.5,
    by_class: bool = False,
    class_name: str = "Socializing, Relaxing, and Leisure",
    class_col: str = "superclass_level0",
) -> tuple[str | None, np.ndarray | None]:
    """
    Decode data for a single subject.

    Args:
        sub: Subject ID
        source_df: Source dataframe containing subject information
        dim: Dimension name (used if by_class=False)
        modality: Data modality ('eeg' or 'meg')
        pick_type: Channel selection type
        cv: Number of cross-validation folds
        inner_n_jobs: Number of parallel jobs for inner decoding
        pca: Whether to apply PCA
        split_thre: Split threshold (used if by_class=False)
        by_class: Whether to decode by class or dimension
        class_name: Class name (used if by_class=True)
        class_col: Class column name (used if by_class=True)

    Returns:
        Tuple of (subject_id, scores_array) or (None, None) if data unavailable
    """
    if modality not in source_df.query("subject == @sub")["session"].values:
        return None, None

    epochs = load_epo(sub, modality)
    if not by_class:
        dim_epo = split_epochs_by_dim(epochs, dim, split_thre)
    else:
        dim_epo = split_epochs_by_class(epochs, class_col, class_name)

    decoder = EpochsDecoder(
        dim_epo,
        pick_type=pick_type,
        random_state=RANDOM_STATE,
    )

    scores: np.ndarray = decoder.sliding_svc(
        scoring="accuracy",
        n_jobs=inner_n_jobs,
        cv=cv,
        pca=pca,
    )
    return sub, scores


def decode_subs_by_dim(
    source_df: pd.DataFrame,
    dim: str,
    modality: str,
    n_jobs: int = 8,
    inner_n_jobs: int = 8,
    cv: int = 10,
    pick_type: str = "all",
    pca: bool = True,
    split_thre: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Decode all subjects using dimension-based splitting.

    Args:
        source_df: Source dataframe with subject information
        dim: Dimension name for splitting
        modality: Data modality ('eeg' or 'meg')
        n_jobs: Number of parallel jobs for subjects
        inner_n_jobs: Number of parallel jobs within each subject's decoding
        cv: Number of cross-validation folds
        pick_type: Channel selection type
        pca: Whether to apply PCA
        split_thre: Split threshold

    Returns:
        Dictionary mapping subject IDs to score arrays
    """
    sub_scores = {}
    subs = source_df["subject"].unique()

    with tqdm_joblib(
        total=len(subs),
        desc=f"Decoding {dim} on {modality}",
        leave=False,
        position=1,
    ):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_decode_one_sub)(
                sub,
                source_df=source_df,
                dim=dim,
                modality=modality,
                pick_type=pick_type,
                cv=cv,
                inner_n_jobs=inner_n_jobs,
                pca=pca,
                split_thre=split_thre,
            )
            for sub in subs
        )
    for sub, scores in results:
        if sub is None:
            continue
        sub_scores[sub] = scores

    return sub_scores


def decode_subs_by_class(
    source_df: pd.DataFrame,
    class_col: str,
    class_name: str,
    modality: str,
    n_jobs: int = 8,
    inner_n_jobs: int = 8,
    cv: int = 10,
    pick_type: str = "all",
    pca: bool = True,
) -> dict[str, np.ndarray]:
    """
    Decode all subjects using class-based splitting.

    Args:
        source_df: Source dataframe with subject information
        class_col: Column name for class classification
        class_name: Name of the target class
        modality: Data modality ('eeg' or 'meg')
        n_jobs: Number of parallel jobs for subjects
        inner_n_jobs: Number of parallel jobs within each subject's decoding
        cv: Number of cross-validation folds
        pick_type: Channel selection type
        pca: Whether to apply PCA

    Returns:
        Dictionary mapping subject IDs to score arrays
    """
    sub_scores = {}
    subs = source_df["subject"].unique()

    with tqdm_joblib(
        total=len(subs),
        desc=f"Decoding {class_name} on {modality}",
        position=1,
        leave=False,
    ):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_decode_one_sub)(
                sub,
                source_df=source_df,
                dim=None,  # Not used for class-based decoding
                modality=modality,
                pick_type=pick_type,
                cv=cv,
                inner_n_jobs=inner_n_jobs,
                pca=pca,
                split_thre=None,  # Not used for class-based decoding
                by_class=True,
                class_name=class_name,
                class_col=class_col,
            )
            for sub in subs
        )
    for sub, scores in results:
        if sub is None:
            continue
        sub_scores[sub] = scores

    return sub_scores


def decode_grid(
    pick: str,
    pca: bool,
    by_class: bool = False,
    dim: str | None = None,
    split_thre: float | None = None,
    class_name: str | None = None,
    class_col: str | None = None,
) -> tuple[dict, plt.Figure]:
    """
    Execute decoding analysis across MEG and EEG modalities.

    Args:
        pick: Channel selection ('all', 'OT', 'O')
        pca: Whether to apply PCA dimensionality reduction
        by_class: If True, decode by class; if False, decode by dimension
        dim: Dimension name (required if by_class=False)
        split_thre: Split threshold (required if by_class=False)
        class_name: Class name (required if by_class=True)
        class_col: Class column name (required if by_class=True)

    Returns:
        Tuple of (results dict, matplotlib figure)

    Raises:
        ValueError: If required parameters are missing based on by_class mode
    """
    res = {}

    if by_class:
        if class_col is None or class_name is None:
            raise ValueError(
                "by_class=True requires class_col and class_name parameters"
            )

        res["eeg"] = decode_subs_by_class(
            cfg.source_df,
            class_col=class_col,
            class_name=class_name,
            modality="eeg",
            n_jobs=N_JOBS,
            cv=10,
            pick_type=pick,
            pca=pca,
        )
        res["meg"] = decode_subs_by_class(
            cfg.source_df,
            class_col=class_col,
            class_name=class_name,
            modality="meg",
            n_jobs=N_JOBS,
            cv=10,
            pick_type=pick,
            pca=pca,
        )
        title = f"Decoding {class_name} Accuracy (%)"
    else:
        if dim is None or split_thre is None:
            raise ValueError("by_class=False requires dim and split_thre parameters")

        res["eeg"] = decode_subs_by_dim(
            cfg.source_df,
            dim=dim,
            modality="eeg",
            n_jobs=N_JOBS,
            cv=10,
            pick_type=pick,
            pca=pca,
            split_thre=split_thre,
        )
        res["meg"] = decode_subs_by_dim(
            cfg.source_df,
            dim=dim,
            modality="meg",
            n_jobs=N_JOBS,
            cv=10,
            pick_type=pick,
            pca=pca,
            split_thre=split_thre,
        )
        title = f"Decoding {dim} Accuracy (%)"

    sample_epo = load_epo("01", "eeg")
    res["timepoints"] = sample_epo.times
    fig = plot_sliding_acc(
        np_Mscore=np.array(list(res["meg"].values())),
        np_Escore=np.array(list(res["eeg"].values())),
        times=res["timepoints"],
        fontsize=FONT_SIZE,
        title=title,
        vmin=PLOT_VMIN,
        vmax=PLOT_VMAX,
        show_individual_accuracy=True,
        color_meg=COLOR_MEG,
        color_eeg=COLOR_EEG,
        verbose=False,
    )
    return res, fig


# %%
if __name__ == "__main__":
    picks = ["all", "OT"]
    dim_thres = np.arange(0.1, 0.6, 0.1).tolist()
    all_classes = {
        "superclass_level0": [
            "Socializing, Relaxing, and Leisure",
            "Sports, Exercise, and Recreation",
            "Household Activities",
            "Eating and drinking Activities",
            "Personal Care",
        ]
    }
    dims = [
        "dangerousness",
        "familiarity",
        "intensity",
        "menace",
        "sociality",
        "transitivity",
    ]

    tasks = []
    for pick in picks:
        for pca in [True, False]:
            for dim in dims:
                for split_thre in dim_thres:
                    basename = f"bydim-{dim}-thre-{split_thre}_soi-{pick}_feat-{'pca' if pca else 'raw'}"
                    tasks.append(
                        {
                            "basename": basename,
                            "feat": "pca" if pca else "raw",
                            "soi": pick,
                            "by": "dim",
                            "dim": dim,
                            "split_thre": split_thre,
                        }
                    )
            for class_col, class_names in all_classes.items():
                for class_name in class_names:
                    basename = f"bycls-{class_col}-{class_name.replace(', ', '_').replace(' ', '')}_soi-{pick}_feat-{'pca' if pca else 'raw'}"
                    tasks.append(
                        {
                            "basename": basename,
                            "feat": "pca" if pca else "raw",
                            "soi": pick,
                            "by": "class",
                            "class_col": class_col,
                            "class_name": class_name,
                        }
                    )
    tasks_df = pd.DataFrame(tasks)
    # %%
    pbar = tqdm(total=len(tasks_df), desc="Decoding tasks")

    for task in tasks_df.itertuples():
        outpkl = SAVE_DIR / "pkls" / f"{task.basename}.pkl"
        outfig = SAVE_DIR / "figs" / f"{task.basename}.png"
        outpkl.parent.mkdir(parents=True, exist_ok=True)
        outfig.parent.mkdir(parents=True, exist_ok=True)
        if outpkl.exists() and outfig.exists():
            pbar.write(f"Skipping existing task: {task.basename}")
            pbar.update(1)
            continue

        try:
            match task.by:
                case "dim":
                    res, fig = decode_grid(
                        by_class=False,
                        pick=task.soi,
                        pca=(task.feat == "pca"),
                        dim=task.dim,
                        split_thre=task.split_thre,
                    )

                case "class":
                    res, fig = decode_grid(
                        by_class=True,
                        pick=task.soi,
                        pca=(task.feat == "pca"),
                        class_name=task.class_name,
                        class_col=task.class_col,
                    )
            with open(outpkl, "wb") as f:
                pkl.dump(res, f)
            fig.savefig(outfig, dpi=300, bbox_inches="tight", transparent=True)
            plt.close(fig)
        except Exception as e:
            pbar.write(f"Error in task {task.basename}: {e}")
            continue

        pbar.update(1)
