# %%
from __future__ import annotations

import contextlib
import io
from typing import Optional

import mne
import numpy as np
from meegkit import dss
from mne.io import BaseRaw

from .bids_loader import BaseLoader


class LineNoiseRunner(BaseLoader):
    def __init__(
        self,
        raw: BaseRaw,
        dtype: str | None = None,
        random_state: int = 42,
    ):
        super().__init__(raw, dtype, random_state)

    def _make_zapline_feature(
        self,
        raw: BaseRaw,
    ) -> np.ndarray:
        """extract zapline feature for line noise influence estimation"""
        data = raw.get_data().T
        data = np.expand_dims(data, axis=2)  # (nsample, nchan, ntrial(1))
        return data

    def _get_n_fline(self, raw: BaseRaw, fline: float = 50.0) -> int:
        """estimate number of components to remove for zapline"""
        low_pass = raw.info['lowpass']
        high_pass = raw.info['highpass']
        n_fline = int((low_pass - high_pass) / fline)
        return n_fline

    def _reconstruct_raw(
        self,
        data: np.ndarray,
        annotations: mne.Annotations,
        info: mne.Info,
    ) -> BaseRaw:
        cleaned_raw = mne.io.RawArray(data, info)
        cleaned_raw.set_annotations(annotations)
        return cleaned_raw

    def _zapline_iter(
        self,
        fline: float = 50.0,
    ) -> BaseRaw:
        """run zapine in all chs for fline influnce all chs"""
        raw = self.raw.copy()
        data = self._make_zapline_feature(raw)
        sfreq = raw.info['sfreq']
        norm_fline = fline / sfreq
        with contextlib.redirect_stdout(io.StringIO()):  # mute for simplicity
            out, _ = dss.dss_line_iter(data, norm_fline, sfreq, nfft=400)
            data_clean = out.T.squeeze()

        return self._reconstruct_raw(
            data_clean,
            self.raw.annotations,
            self.raw.info,
        )

    def _zapline(
        self,
        fline: float = 50.0,
    ) -> BaseRaw:
        'run zapline in all chs for fline influnce all chs'
        raw = self.raw.copy()
        data = self._make_zapline_feature(raw)
        sfreq = raw.info['sfreq']
        norm_fline = fline / sfreq
        nremove = self.get_n_fline(raw, fline)
        with contextlib.redirect_stdout(io.StringIO()):  # mute for simplicity
            out, _ = dss.dss_line(
                data,
                fline=norm_fline,
                sfreq=raw.info['sfreq'],
                nremove=nremove,
                blocksize=1000,
                show=False,
            )
            data_clean = out.T.squeeze()

        return self._reconstruct_raw(
            data_clean,
            self.raw.annotations,
            self.raw.info,
        )

    def run(
        self,
        fline: float = 50.0,
        find_in: list | None = None,
    ) -> BaseRaw:
        self.raw = self._pick_chs(self.raw, find_in)
        match self.dtype:
            case 'meg':
                return self._zapline(fline)
            case 'eeg':
                return self._zapline_iter(fline)
            case _:
                raise ValueError("datatype must be 'eeg' or 'meg'")
