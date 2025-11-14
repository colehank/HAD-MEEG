# %%
from __future__ import annotations

from typing import Optional

from mne.io import BaseRaw
from mne.preprocessing import (
    find_bad_channels_maxwell,
)
from pyprep.find_noisy_channels import NoisyChannels

from .bids_loader import BaseLoader


class BadChsRunner(BaseLoader):
    def __init__(
        self,
        raw: BaseRaw,
        dtype: str | None = None,
        random_state: int = 42,
    ):
        super().__init__(raw, dtype, random_state)

    def _handle_meg(
        self,
        origin=(0., 0., 0.04),
        find_in: list | None = None,
    ) -> BaseRaw:
        raw = self.raw.copy()  # for return
        _raw = self.raw.copy()  # for detection
        # It seems like MNE require CTF data in 0-composentation
        # to run the find_bad_channels_maxwell fuc.
        # To protect raw influenced by multi-composentation,
        # we use a copy here for bad ch detection
        if _raw.compensation_grade != 0:
            _raw.apply_gradient_compensation(0)
        _raw = self._pick_chs(_raw, find_in)

        auto_noisy_chs, auto_flat_chs, auto_score = find_bad_channels_maxwell(
            raw=_raw,
            return_scores=True,
            origin=origin,
            cross_talk=None,
            calibration=None,
        )
        self.bads = list(set(auto_noisy_chs + auto_flat_chs))
        raw.info['bads'].extend(self.bads)
        return raw

    def _handle_eeg(
        self,
        find_in: list | None = None,
    ) -> BaseRaw:
        raw = self.raw.copy()
        raw = self._pick_chs(raw, find_in)

        finder = NoisyChannels(raw, random_state=self.random_state)
        finder.find_bad_by_correlation()
        finder.find_bad_by_deviation()
        finder.find_bad_by_ransac()

        self.bads = finder.get_bads()
        raw.info['bads'].extend(self.bads)
        return raw

    def run(
        self,
        find_in: list | None = None,
        fix: bool = True,
        reset_bads: bool = True,
        origin=(0., 0., 0.04),
    ) -> BaseRaw:

        match self.dtype:
            case 'meg':
                clean_raw = self._handle_meg(origin, find_in)
            case 'eeg':
                clean_raw = self._handle_eeg(find_in)

        if fix:
            clean_raw.load_data()
            clean_raw.interpolate_bads(
                reset_bads=reset_bads,
                method=dict(meg='MNE', eeg='spline'),
            )
        return clean_raw
