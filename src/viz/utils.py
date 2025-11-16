from __future__ import annotations

from typing import Union

import mne
import numpy as np


def get_2d_pos(
    raw: mne.io.BaseRaw,
    picks: list[int] | str = None,
) -> np.ndarray:
    from mne.channels.layout import _find_topomap_coords

    raw = raw.copy()
    if picks is not None:
        raw.pick(picks)
    xy = _find_topomap_coords(raw.info, picks)
    center = xy.mean(axis=0)
    layout_pos = xy - center
    return layout_pos


def plot_2d_topo(
    data: np.ndarray,
    raw: mne.io.BaseRaw,
    picks: list[int] | str = None,
    **kwargs,
):
    if picks is None:
        picks = raw.info['ch_names']
    mne.viz.plot_topomap(
        data=data,
        pos=get_2d_pos(raw, picks=picks),
        **kwargs,
    )
