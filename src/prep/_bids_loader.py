from __future__ import annotations

from typing import Literal

from loguru import logger
from mne.io import BaseRaw

DType = Literal['eeg', 'meg']


class BaseLoader:
    def __init__(
        self,
        raw: BaseRaw,
        dtype: DType | None = None,
        random_state: int = 42,
    ):
        if dtype is None:
            dtype = self._infer_dtype(raw)
            if dtype is None or dtype not in ['eeg', 'meg']:
                raise ValueError('datatype cannot be inferred or is invalid.')

        assert dtype in ['eeg', 'meg'], "datatype must be 'eeg' or 'meg'"

        logger.trace(f'Initialized BaseLoader with dtype={dtype}')
        self.raw = raw
        self.dtype: DType = dtype
        self.random_state = random_state

    def _infer_dtype(self, raw) -> str:
        ch_types = set(raw.get_channel_types())
        if 'eeg' in ch_types:
            return 'eeg'
        if any(t in ch_types for t in ('mag', 'grad', 'planar1', 'planar2')):
            return 'meg'
        else:
            return None

    def _pick_chs(
        self,
        raw: BaseRaw,
        picks: list | None = None,
    ) -> BaseRaw:
        if picks is None:
            picks = (
                [
                    'mag',
                    'grad',
                    'planar1',
                    'planar2',
                ]
                if self.dtype == 'meg'
                else ['eeg']
            )
            raw.pick(picks)
        return raw
