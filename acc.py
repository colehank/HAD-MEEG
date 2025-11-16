# %%
from __future__ import annotations

import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm

from src import DataConfig

config = DataConfig()
ROOT = config.bids_root
DERI_EV_DIR = op.join(config.derivatives_root, 'detailed_events')

SAVE_DIR = '../HAD-MEEG_results'
FIG_DIR = f'{SAVE_DIR}/figs'
os.makedirs(FIG_DIR, exist_ok=True)

FONT_SIZE = 18
FONT_PATH = 'resources/Helvetica.ttc'

fm.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()

colormap = 'Spectral'
cmap_ = plt.get_cmap(colormap, 10)
colors = [cmap_(i / (10 - 1)) for i in range(10)]
color_meg = colors[2]
color_eeg = colors[-3]
# %%


def get_files(root_dir: str) -> dict[str, str]:
    """Get a dictionary mapping subject IDs to file paths in the specified directory.

    Parameters
    ----------
    root_dir : str
        The directory containing the CSV files.

    Returns
    -------
    dict
        A dictionary mapping subject IDs to file paths.
    """
    files: dict[str, str] = {}

    for filename in sorted(os.listdir(root_dir)):
        if filename.endswith('.csv'):
            subject_id = filename.split('_')[0][-2:]
            files[subject_id] = op.join(root_dir, filename)

    return files


def extract_sub_accuracy(file_path: str) -> list[dict[str, object]]:
    """Extract accuracy per session for a given subject file.

    Parameters
    ----------
    file_path : str
        Path to the subject's data file.

    Returns
    -------
    list of dict
        A list of records with keys: 'sub', 'session', 'run', 'accuracy'.
    """
    event_data = pd.read_csv(file_path)
    subject_accuracy: list[dict[str, object]] = []

    for sess in event_data['session'].unique():
        sess_data = event_data[event_data['session'] == sess]
        runs = sess_data['run'].unique()

        for run in runs:
            run_data = sess_data[sess_data['run'] == run]
            correct_responses = run_data[run_data['resp_is_right']]

            if len(run_data) > 0:
                accuracy = len(correct_responses) / len(run_data)
            else:
                accuracy = np.nan

            subject_accuracy.append(
                {
                    'sub': file_path.split('/')[-1].split('_')[0][-2:],
                    'session': sess,
                    'run': run,
                    'accuracy': accuracy,
                },
            )

    return subject_accuracy


def extract_accuracy(files_dict: dict[str, str]) -> pd.DataFrame:
    """Extract accuracies for all subjects given a mapping from subject ID to file path."""
    accuracies: list[pd.DataFrame] = []

    for subject_id in files_dict:
        acc = pd.DataFrame(extract_sub_accuracy(files_dict[subject_id]))
        accuracies.append(acc)

    return pd.concat(accuracies, ignore_index=True)


def plot_accuracy_violin_box(
    df: pd.DataFrame,
    font_size: int = FONT_SIZE,
    color_meg: str = '#1f77b4',
    color_eeg: str = '#ff7f0e',
) -> plt.Figure:
    """Plot subject accuracy (MEG/EEG) as a split violin plot + boxplot."""
    df_box = df[['sub', 'acc']].copy()

    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)

    sns.violinplot(
        data=df,
        x='sub',
        y='acc',
        hue='modality',
        inner=None,
        ax=ax,
        native_scale=True,
        linewidth=0,
        density_norm='width',
        split=True,
        palette={'MEG': color_meg, 'EEG': color_eeg},
    )

    sns.boxplot(
        data=df_box,
        x='sub',
        y='acc',
        showcaps=True,
        boxprops={
            'facecolor': 'dimgray',
            'edgecolor': 'dimgray',
            'alpha': 0.6,
            'linewidth': 0.8,
        },
        showfliers=False,
        whiskerprops={'linewidth': 1.0, 'color': 'dimgray'},
        medianprops={'color': 'lightgoldenrodyellow', 'linewidth': 2.2},
        width=0.15,
        ax=ax,
    )

    ax.set_xlabel('Subject', fontsize=font_size)
    ax.set_ylabel('Recognition Accuracy', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size - 2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim(0.47, 1)
    ax.set_yticks([0.6, 0.8, 1.0])

    ax.legend(
        loc='lower right',
        frameon=False,
        fontsize=font_size - 4,
        bbox_to_anchor=(1.02, 0),
        handlelength=1,
    )

    fig.subplots_adjust(bottom=0.15, left=0.08, right=0.98, top=0.9)
    plt.tight_layout()
    plt.show()
    return fig


# %%
if __name__ == '__main__':
    all_events = get_files(DERI_EV_DIR)
    acc = extract_accuracy(all_events)

    acc.columns = [c.strip().lower() for c in acc.columns]
    sub_col = 'sub' if 'sub' in acc.columns else 'subject'
    acc_col = 'accuracy' if 'accuracy' in acc.columns else 'acc'
    mod_col = 'session' if 'session' in acc.columns else 'modality'
    df = acc[[sub_col, mod_col, acc_col]].copy()
    df = df.rename(
        columns={sub_col: 'sub', mod_col: 'modality', acc_col: 'acc'},
    )
    df['modality'] = df['modality'].astype(str).str.upper().str.strip()
    df['sub'] = df['sub'].astype(str)
    df = df[df['modality'].isin(['EEG', 'MEG'])]
    fig = plot_accuracy_violin_box(df, FONT_SIZE, color_meg, color_eeg)

    acc.to_csv(op.join(SAVE_DIR, 'accuracy.csv'), index=False)
    fig.savefig(
        op.join(FIG_DIR, 'accuracy.svg'),
        dpi=300,
        bbox_inches='tight',
        transparent=True,
    )
    fig.savefig(
        op.join(FIG_DIR, 'accuracy.png'),
        dpi=300,
        bbox_inches='tight',
        transparent=True,
    )
# %%
