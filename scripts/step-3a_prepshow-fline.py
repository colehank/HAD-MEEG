"""Visualize power spectral density (PSD) comparison between raw and preprocessed MEG/EEG data.

This script generates a 2x2 grid plot comparing PSD across all channels for:
- Raw MEG data vs. preprocessed MEG data
- Raw EEG data vs. preprocessed EEG data
"""

# %%
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mne
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from mne.io import BaseRaw
from mne_bids import read_raw_bids
from loguru import logger
from src import DataConfig, PlotConfig

# Configuration
cfg_data = DataConfig()
cfg_plot = PlotConfig()

subject = "01"
session = "action"
run = "01"

# Output configuration
save_dir = cfg_data.results_root / "prepshow-fline"
save_dir.mkdir(parents=True, exist_ok=True)

# Visualization constants
N_COLORS = 290  # Number of colors in colormap (should exceed number of channels)
ALPHA = 0.4  # Transparency for channel lines
LINEWIDTH = 0.1  # Line width for channel plots
FREQ_MIN = 0.1  # Minimum frequency for PSD plot (Hz)
FREQ_MAX = 110  # Maximum frequency for PSD plot (Hz)
PSD_MIN = -10  # Minimum PSD value for y-axis (dB)
PSD_MAX = 40  # Maximum PSD value for y-axis (dB)
FONTSIZE = cfg_plot.font_size
FONT_PATH = cfg_plot.font_path
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
# Setup font


# %%
# Utility functions
def align_raw(raw: BaseRaw, params: dict[str, float]) -> BaseRaw:
    """Align raw data to match preprocessing parameters.

    Args:
        raw: Raw MEG/EEG data object.
        params: Dictionary containing alignment parameters:
            - sfreq: Target sampling frequency (Hz)
            - hfreq: High-pass filter frequency (Hz)
            - lfreq: Low-pass filter frequency (Hz)

    Returns:
        Aligned raw data with matched sampling rate and frequency band.
    """
    raw = raw.copy().load_data()

    # Select appropriate channel type
    dtype = "meg" if "meg" in raw else "eeg" if "eeg" in raw else None
    if dtype == "meg":
        raw = raw.pick_types(meg="mag", ref_meg=False)
    elif dtype == "eeg":
        raw = raw.pick_types(eeg=True)

    # Apply resampling and filtering
    raw.resample(params["sfreq"])
    raw.filter(params["hfreq"], params["lfreq"])

    return raw


def plot_psd_all_channels(
    psd: mne.time_frequency.Spectrum,
    ax: plt.Axes,
    colors: list,
    alpha: float = 0.15,
    linewidth: float = 0.5,
) -> None:
    """Plot PSD for all channels with whitening normalization.

    Args:
        psd: Computed power spectral density object.
        ax: Matplotlib axes to plot on.
        colors: List of colors for each channel.
        alpha: Transparency level for channel lines (0-1).
        linewidth: Width of channel lines.
    """
    psd_data = psd.get_data()
    freqs = psd.freqs

    # Convert to dB scale
    psd_data_db = 10 * np.log10(psd_data)

    # Whiten by removing channel-wise mean
    psd_data_db_mean = np.mean(psd_data_db, axis=1, keepdims=True)
    psd_data_db -= psd_data_db_mean

    # Plot each channel
    for i in range(psd_data_db.shape[0]):
        ax.plot(
            freqs,
            psd_data_db[i, :],
            color=colors[i],
            alpha=alpha,
            linewidth=linewidth,
        )


# %%
# Main execution
if __name__ == "__main__":
    # Load raw and preprocessed data
    run_idx = int(run) - 1
    raw_meg = read_raw_bids(cfg_data.source[subject]["meg"][run_idx])
    raw_eeg = read_raw_bids(cfg_data.source[subject]["eeg"][run_idx])
    logger.info("Loaded raw MEG and EEG data.")
    clean_meg = mne.io.read_raw(cfg_data.preprocessed[subject]["meg"][run_idx])
    clean_eeg = mne.io.read_raw(cfg_data.preprocessed[subject]["eeg"][run_idx])

    # Align raw data to match preprocessing parameters
    align_params = {
        "lfreq": clean_meg.info["lowpass"],
        "hfreq": clean_meg.info["highpass"],
        "sfreq": clean_meg.info["sfreq"],
    }
    logger.info("Aligned raw MEG and EEG data to match preprocessing parameters.")
    raw_meg = align_raw(raw_meg, align_params)
    raw_eeg = align_raw(raw_eeg, align_params)

    # Compute power spectral density
    raw_psd_meg = raw_meg.compute_psd()
    raw_psd_eeg = raw_eeg.compute_psd()
    clean_psd_meg = clean_meg.compute_psd()
    clean_psd_eeg = clean_eeg.compute_psd()

    # Setup colormaps for visualization
    cmap_raw = cm.get_cmap("Spectral_r", N_COLORS)
    cmap_clean = cm.get_cmap("Spectral", N_COLORS)
    colors_raw = [cmap_raw(i / (N_COLORS - 1)) for i in range(N_COLORS)]
    colors_clean = [cmap_clean(i / (N_COLORS - 1)) for i in range(N_COLORS)]

    # Create custom formatter for x-axis ticks
    freq_formatter = ticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x == 0.1 else f"{int(x)}"
    )
    logger.info("Plotting.")
    # Create figure with 2x2 grid layout
    plt.close("all")
    fig = plt.figure(figsize=(12, 3), dpi=300)

    gs = GridSpec(2, 2)
    ax3 = fig.add_subplot(gs[1:, :1])  # Bottom-left: MEG preprocessed
    ax4 = fig.add_subplot(gs[1:, 1:])  # Bottom-right: EEG preprocessed
    ax1 = fig.add_subplot(gs[:1, :1], sharex=ax3, sharey=ax3)  # Top-left: MEG raw
    ax2 = fig.add_subplot(gs[:1, 1:], sharex=ax4, sharey=ax4)  # Top-right: EEG raw

    # Plot PSD for all channels (raw data uses reversed colormap, clean uses normal)
    plot_psd_all_channels(
        raw_psd_meg, ax=ax1, colors=colors_raw, alpha=ALPHA, linewidth=LINEWIDTH
    )
    plot_psd_all_channels(
        raw_psd_eeg, ax=ax2, colors=colors_raw, alpha=ALPHA, linewidth=LINEWIDTH
    )
    plot_psd_all_channels(
        clean_psd_meg, ax=ax3, colors=colors_clean, alpha=ALPHA, linewidth=LINEWIDTH
    )
    plot_psd_all_channels(
        clean_psd_eeg, ax=ax4, colors=colors_clean, alpha=ALPHA, linewidth=LINEWIDTH
    )

    # Configure axes appearance
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_xticks([0.1, 20, 40, 60, 80, 100])
        ax.set_ylim(PSD_MIN, PSD_MAX)
        ax.xaxis.set_major_formatter(freq_formatter)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=FONTSIZE - 2)

        # Hide x-axis for top row
        if ax in [ax1, ax2]:
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)

        # Add x-label for bottom row
        if ax in [ax3, ax4]:
            ax.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE)

        # Add y-labels
        if ax in [ax1, ax3]:
            ax.set_ylabel(r"fT$^2$/Hz (dB)", fontsize=FONTSIZE)
        if ax in [ax2, ax4]:
            ax.set_ylabel(r"$\mu$V$^2$/Hz (dB)", labelpad=0, fontsize=FONTSIZE)

    # Add panel labels
    label_font = FONTSIZE + 2
    label_params = {
        "transform": ax1.transAxes,
        "fontsize": label_font,
        "verticalalignment": "top",
        "horizontalalignment": "right",
    }

    ax1.text(0.95, 0.95, "MEG-raw", **label_params)
    ax2.text(0.95, 0.95, "EEG-raw", **label_params)
    ax3.text(0.95, 0.95, "MEG-cleaned", **label_params)
    ax4.text(0.95, 0.95, "EEG-cleaned", **label_params)

    # Save figure
    plt.show()
    fig.savefig(
        save_dir / "fline.svg",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        save_dir / "fline.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

# %%
