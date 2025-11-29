import mne
import numpy as np
import matplotlib.pyplot as plt


def get_soi_picks(
    inst: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
    soi: str,
    plot: bool = False,
) -> list[str]:
    """Get the picks of the SOI.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.Epochs | mne.Evoked
        The data object.
    soi : str
        The SOI. avaiable options: 'F', 'C', 'P', 'O', 'T', 'all'.
    plot : bool, optional
        Plot the sensors.
    """
    inst_ = inst.copy()
    ch_names = inst_.ch_names
    n_channels = len(ch_names)

    if soi == "all":
        picks = list(range(n_channels))
    else:
        picks = [
            i for i, ch in enumerate(ch_names) if any([s_type in ch for s_type in soi])
        ]
    if plot:
        inst_.pick(picks)
        inst_.plot_sensors()
    return picks


def _get_meg_pos(epo):
    ch_name_picks = mne.pick_channels_regexp(epo.ch_names, regexp="M[LRZ]...")
    type_picks = mne.pick_types(epo.info, meg=True)
    picks = np.intersect1d(ch_name_picks, type_picks)
    intercorr_chn = [epo.ch_names[idx][:5] for idx in picks]
    original_meg_layout = mne.channels.find_layout(epo.info, ch_type="meg")
    exclude_list = [x for x in original_meg_layout.names if x[:5] not in intercorr_chn]
    meg_layout = mne.channels.find_layout(epo.info, ch_type="meg", exclude=exclude_list)
    pos = meg_layout.pos[:, :2]
    layout_pos = pos - pos.mean(axis=0)
    return layout_pos


def plot_sensors(
    inst: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
    sois: list[str] = ["O", "P", "C", "F", "T"],
) -> None:
    """Plot the sensors of the SOI.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.Epochs | mne.Evoked
        The data object.
    sois : list[str]
        The SOIs.
    """
    full_name = {
        "F": "Frontal",
        "C": "Central",
        "P": "Parietal",
        "O": "Occipital",
        "T": "Temporal",
    }
    groups = {}
    for soi in sois:
        picks = get_soi_picks(inst, soi)
        groups[soi] = picks
    ch_groups = list(groups.values())
    colors = plt.cm.get_cmap("Set2", len(ch_groups))

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for soi, color in zip(sois, colors(range(len(ch_groups)))):
        ax.scatter([], [], color=color, label=full_name[soi])
    ax.legend(loc="best", framealpha=0.9)

    mne.viz.plot_sensors(
        info=inst.info,
        ch_groups=ch_groups,
        linewidth=1,
        pointsize=40,
        cmap=colors,
        axes=ax,
        to_sphere=True,
        show=False,
    )

    return fig
