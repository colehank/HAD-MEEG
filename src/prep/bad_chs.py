# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import (
    find_bad_channels_maxwell,
)
from pyprep.find_noisy_channels import NoisyChannels

from ._bids_loader import BaseLoader


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
        origin=(0.0, 0.0, 0.04),
        find_in: list | None = None,
    ) -> BaseRaw:
        raw = self.raw.copy()  # for return
        _raw = self.raw.copy()  # for detection
        # It seems like MNE require CTF data in 0-composentation
        # to run the find_bad_channels_maxwell fuc.
        # To protect raw influenced by multi-composentation,
        # we use a copy here for bad ch detection
        if _raw.compensation_grade != 0:
            logger.trace(
                f'CTF data has compensation grade {_raw.compensation_grade},'
                f'applying 0-compensation to its copy for bad channel detection.',
            )
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
        logger.trace(
            f'Detected {len(self.bads)} bad MEG channels',
        )
        return raw

    def _handle_eeg(
        self,
        find_in: list | None = None,
    ) -> BaseRaw:
        logger.trace('Detecting bad EEG channels...')
        raw = self.raw.copy()
        raw = self._pick_chs(raw, find_in)

        finder = NoisyChannels(raw, random_state=self.random_state)
        finder.find_bad_by_correlation()
        finder.find_bad_by_deviation()
        finder.find_bad_by_ransac()

        self.bads = finder.get_bads()
        raw.info['bads'].extend(self.bads)
        logger.trace(
            f'Detected {len(self.bads)} bad EEG channels',
        )
        return raw

    def run(
        self,
        find_in: list | None = None,
        fix: bool = True,
        reset_bads: bool = True,
        origin=(0.0, 0.0, 0.04),
        save_deriv: bool = True,
        fname: str | None = None,
    ) -> BaseRaw:
        logger.info('bad channels detecting & fixing')
        match self.dtype:
            case 'meg':
                clean_raw = self._handle_meg(origin, find_in)
            case 'eeg':
                clean_raw = self._handle_eeg(find_in)
            case _:
                raise ValueError("datatype must be 'eeg' or 'meg'")
        if fix:
            clean_raw.load_data()
            clean_raw.interpolate_bads(
                reset_bads=reset_bads,
                method=dict(meg='MNE', eeg='spline'),
            )

        if save_deriv:
            if fname is None:
                raise ValueError(
                    'Please provide a filename to save the derivative.',
                )
            fname = Path(f'{fname}_desc-badchs_{self.dtype}.tsv')
            os.makedirs(fname.parent, exist_ok=True)
            chs = clean_raw.ch_names
            status = ['good'] * len(chs)
            status_desc = ['fixed' if ch in self.bads else 'n/a' for ch in chs]

            df = pd.DataFrame(
                {
                    'name': chs,
                    'type': [self.dtype] * len(chs),
                    'status': status,
                    'status_description': status_desc,
                },
            )
            df.to_csv(
                fname,
                sep='\t',
                index=False,
                encoding='utf-8',
                na_rep='n/a',
            )

            fname_json = fname.with_suffix('.json')
            meta = {
                'name': "Channels' name",
                'type': 'Channel type, e.g., EEG, MEG',
                'status': 'Channel status, good or bad',
                'status_description': 'Description of the channel status, e.g., fixed if interpolated',
            }
            with open(fname_json, 'w') as f:
                json.dump(meta, f, indent=4)

            logger.trace(f'Saved bad channel annotated raw to {fname}')
            logger.trace(f'Saved sidecar json to {fname_json}')

        return clean_raw
