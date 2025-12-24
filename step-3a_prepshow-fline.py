# %%
from mne_bids import read_raw_bids
import mne
from mne.io import BaseRaw
import numpy as np
from src import DataConfig
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.ticker as ticker
from pathlib import Path

cfg = DataConfig()
SUB = "01"
SES = "action"
RUN = "01"

ROOT = cfg.bids_root

SAVE_DIR = cfg.results_root / "prepshow-fline"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FONTSIZE = 12
COLORMAP = "Spectral"
NCOLOR = 290  # > nchan
FONT_PATH = Path("resources") / "Helvetica.ttc"

fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()


# %%
def align_raw(
    raw: BaseRaw,
    paras: dict[str, float],
) -> BaseRaw:
    raw = raw.copy().load_data()

    dtype = "meg" if "meg" in raw else "eeg" if "eeg" in raw else None
    if dtype == "meg":
        raw = raw.pick_types(meg="mag", ref_meg=False)
    elif dtype == "eeg":
        raw = raw.pick_types(eeg=True)

    raw.resample(paras["sfreq"])
    raw.filter(paras["hfreq"], paras["lfreq"])

    return raw


def plot_psd_all_channels(
    psd: mne.time_frequency.Spectrum,
    ax: plt.Axes,
    colors: list,
    alpha: float = 0.15,
    linewidth: float = 0.5,
):
    psd_data = psd.get_data()
    freqs = psd.freqs

    psd_data_db = 10 * np.log10(psd_data)  # PSD -> dB (10 * log10(PSD))
    psd_data_db_mean = np.mean(psd_data_db, axis=1, keepdims=True)
    psd_data_db -= psd_data_db_mean  # whitening

    for i in range(psd_data_db.shape[0]):
        ax.plot(
            freqs, psd_data_db[i, :], color=colors[i], alpha=alpha, linewidth=linewidth
        )


# %%
if __name__ == "__main__":
    rawMEG = read_raw_bids(cfg.source[SUB]["meg"][int(RUN) - 1])
    rawEEG = read_raw_bids(cfg.source[SUB]["eeg"][int(RUN) - 1])

    cleanMEG = mne.io.read_raw(cfg.preprocessed[SUB]["meg"][int(RUN) - 1])
    cleanEEG = mne.io.read_raw(cfg.preprocessed[SUB]["eeg"][int(RUN) - 1])
    # %% align MEG/EEG's sampling frequency, band pass for comparison and compute PSD
    paras = {
        "lfreq": cleanMEG.info["lowpass"],
        "hfreq": cleanMEG.info["highpass"],
        "sfreq": cleanMEG.info["sfreq"],
    }

    rawMEG = align_raw(rawMEG, paras)
    rawEEG = align_raw(rawEEG, paras)

    raw_psdMEG = rawMEG.compute_psd()
    raw_psdEEG = rawEEG.compute_psd()

    clean_psdMEG = cleanMEG.compute_psd()
    clean_psdEEG = cleanEEG.compute_psd()
    # %% plot PSD
    alpha = 0.4
    linewidth = 0.1
    cmap_raw = cm.get_cmap("Spectral_r", NCOLOR)
    cmap_clean = cm.get_cmap("Spectral", NCOLOR)
    colors_raw = [cmap_raw(i / (NCOLOR - 1)) for i in range(NCOLOR)]
    colors_clean = [cmap_clean(i / (NCOLOR - 1)) for i in range(NCOLOR)]

    formatter = ticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x == 0.1 else f"{int(x)}"
    )

    plt.close("all")
    fig = plt.figure(figsize=(12, 3), dpi=300)

    gs = GridSpec(2, 2)
    ax3 = fig.add_subplot(gs[1:, :1])  # MEG cleaned
    ax4 = fig.add_subplot(gs[1:, 1:])  # EEG cleaned
    ax1 = fig.add_subplot(gs[:1, :1], sharex=ax3, sharey=ax3)  # MEG raw
    ax2 = fig.add_subplot(gs[:1, 1:], sharex=ax4, sharey=ax4)  # EEG raw

    plot_psd_all_channels(
        raw_psdMEG, ax=ax1, colors=colors_raw, alpha=alpha, linewidth=linewidth
    )
    plot_psd_all_channels(
        raw_psdEEG, ax=ax2, colors=colors_raw, alpha=alpha, linewidth=linewidth
    )
    plot_psd_all_channels(
        clean_psdMEG, ax=ax3, colors=colors_raw, alpha=alpha, linewidth=linewidth
    )
    plot_psd_all_channels(
        clean_psdEEG, ax=ax4, colors=colors_raw, alpha=alpha, linewidth=linewidth
    )

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0.1, 110)
        ax.set_xticks([0.1, 20, 40, 60, 80, 100])
        ax.set_ylim(-10, 40)
        ax.xaxis.set_major_formatter(formatter)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=FONTSIZE - 2)

        if ax in [ax1, ax2]:
            ax.spines["bottom"].set_visible(False)
            ax.get_xaxis().set_visible(False)
        if ax in [ax3, ax4]:
            ax.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE)
        if ax in [ax1, ax3]:
            ax.set_ylabel(r"fT$^2$/Hz (dB)", fontsize=FONTSIZE)
        if ax in [ax2, ax4]:
            ax.set_ylabel(r"$\mu$V$^2$/Hz (dB)", labelpad=0, fontsize=FONTSIZE)

    label_font = FONTSIZE + 2
    ax1.text(
        0.95,
        0.95,
        "MEG-raw",
        transform=ax1.transAxes,
        fontsize=label_font,
        verticalalignment="top",
        horizontalalignment="right",
    )
    ax2.text(
        0.95,
        0.95,
        "EEG-raw",
        transform=ax2.transAxes,
        fontsize=label_font,
        verticalalignment="top",
        horizontalalignment="right",
    )
    ax3.text(
        0.95,
        0.95,
        "MEG-cleaned",
        transform=ax3.transAxes,
        fontsize=label_font,
        verticalalignment="top",
        horizontalalignment="right",
    )
    ax4.text(
        0.95,
        0.95,
        "EEG-cleaned",
        transform=ax4.transAxes,
        fontsize=label_font,
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.show()
    fig.savefig(f"{SAVE_DIR}/fline.svg", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(f"{SAVE_DIR}/fline.png", dpi=300, bbox_inches="tight", transparent=True)
# %%
