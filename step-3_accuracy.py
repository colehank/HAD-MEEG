from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm

from src import DataConfig


# === Paths & constants ===
config = DataConfig()
DERI_EV_DIR = Path(config.derivatives_root) / "detailed_events"

SAVE_DIR = Path("../HAD-MEEG_results")
FIG_DIR = SAVE_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT_SIZE: int = 18
FONT_PATH = Path("resources/Helvetica.ttc")

COLORMAP = "Spectral"
cmap = plt.get_cmap(COLORMAP, 10)
colors = [cmap(i / (10 - 1)) for i in range(10)]
COLOR_MEG = colors[2]
COLOR_EEG = colors[-3]


# === Matplotlib / font setup ===
fm.fontManager.addfont(str(FONT_PATH))
plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()


def get_files(root_dir: Path) -> dict[str, Path]:
    """
    Build a mapping from subject IDs to CSV file paths in the given directory.

    Parameters
    ----------
    root_dir : Path
        Directory containing the CSV files.

    Returns
    -------
    dict[str, Path]
        Mapping from subject ID to corresponding CSV path.
    """
    files: dict[str, Path] = {}

    for path in sorted(root_dir.glob("*.csv")):
        # Assumes filenames like "sub-01_*.csv" → subject_id "01"
        subject_id = path.name.split("_")[0][-2:]
        files[subject_id] = path

    return files


def extract_sub_accuracy(file_path: Path) -> list[dict[str, object]]:
    """
    Extract accuracy per session and run for a single subject.

    Parameters
    ----------
    file_path : Path
        Path to the subject's CSV file.

    Returns
    -------
    list[dict[str, object]]
        Records with keys: 'sub', 'session', 'run', 'accuracy'.
    """
    event_data = pd.read_csv(file_path)
    subject_accuracy: list[dict[str, object]] = []

    for sess in event_data["session"].unique():
        sess_data = event_data[event_data["session"] == sess]

        for run in sess_data["run"].unique():
            run_data = sess_data[sess_data["run"] == run]
            correct_responses = run_data[run_data["resp_is_right"]]

            if len(run_data) > 0:
                accuracy = len(correct_responses) / len(run_data)
            else:
                accuracy = np.nan

            subject_accuracy.append(
                {
                    "sub": file_path.stem.split("_")[0][-2:],
                    "session": sess,
                    "run": run,
                    "accuracy": accuracy,
                },
            )

    return subject_accuracy


def extract_accuracy(files_dict: dict[str, Path]) -> pd.DataFrame:
    """
    Extract accuracy for all subjects in the given mapping.

    Parameters
    ----------
    files_dict : dict[str, Path]
        Mapping from subject ID to CSV file path.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'sub', 'session', 'run', 'accuracy'.
    """
    all_acc: list[pd.DataFrame] = []

    for file_path in files_dict.values():
        acc = pd.DataFrame(extract_sub_accuracy(file_path))
        all_acc.append(acc)

    return pd.concat(all_acc, ignore_index=True)


def plot_accuracy_violin_box(
    df: pd.DataFrame,
    font_size: int = FONT_SIZE,
    color_meg: str | tuple = COLOR_MEG,
    color_eeg: str | tuple = COLOR_EEG,
) -> plt.Figure:
    """
    Plot subject recognition accuracy (MEG/EEG) as split violin + boxplot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'sub', 'modality', 'acc'.
    font_size : int
        Base font size for labels and ticks.
    color_meg : color-like
        Color for MEG violins.
    color_eeg : color-like
        Color for EEG violins.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    df_box = df[["sub", "acc"]].copy()

    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)

    sns.violinplot(
        data=df,
        x="sub",
        y="acc",
        hue="modality",
        inner=None,
        ax=ax,
        native_scale=True,
        linewidth=0,
        density_norm="width",
        split=True,
        palette={"MEG": color_meg, "EEG": color_eeg},
    )

    sns.boxplot(
        data=df_box,
        x="sub",
        y="acc",
        showcaps=True,
        boxprops={
            "facecolor": "dimgray",
            "edgecolor": "dimgray",
            "alpha": 0.6,
            "linewidth": 0.8,
        },
        showfliers=False,
        whiskerprops={"linewidth": 1.0, "color": "dimgray"},
        medianprops={"color": "lightgoldenrodyellow", "linewidth": 2.2},
        width=0.15,
        ax=ax,
    )

    ax.set_xlabel("Subject", fontsize=font_size)
    ax.set_ylabel("Recognition Accuracy", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size - 2)

    # Clean up spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Y axis limits and ticks
    ax.set_ylim(0.47, 1.0)
    ax.set_yticks([0.6, 0.8, 1.0])

    # Legend
    ax.legend(
        ncols=2,
        loc="lower right",
        frameon=False,
        fontsize=font_size - 4,
        bbox_to_anchor=(1.01, -0.07),
        handlelength=1.2,
        columnspacing=0.5,
        handletextpad=0.3,
    )

    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    all_events = get_files(DERI_EV_DIR)
    acc = extract_accuracy(all_events)

    # Normalize column names
    acc.columns = [c.strip().lower() for c in acc.columns]
    sub_col = "sub" if "sub" in acc.columns else "subject"
    acc_col = "accuracy" if "accuracy" in acc.columns else "acc"
    mod_col = "session" if "session" in acc.columns else "modality"

    df = acc[[sub_col, mod_col, acc_col]].copy()
    df = df.rename(
        columns={sub_col: "sub", mod_col: "modality", acc_col: "acc"},
    )

    df["modality"] = df["modality"].astype(str).str.upper().str.strip()
    df["sub"] = df["sub"].astype(str)
    df = df[df["modality"].isin(["EEG", "MEG"])]

    fig = plot_accuracy_violin_box(df)

    acc.to_csv(SAVE_DIR / "accuracy.csv", index=False)

    fig.savefig(
        FIG_DIR / "accuracy.svg",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        FIG_DIR / "accuracy.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
