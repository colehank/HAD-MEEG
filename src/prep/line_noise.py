# %%
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from loguru import logger
from meegkit import dss
from mne.io import BaseRaw

from ._bids_loader import BaseLoader


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

    def _reconstruct_raw(
        self,
        data: np.ndarray,
        annotations: mne.Annotations,
        info: mne.Info,
    ) -> BaseRaw:
        """reconstruct cleaned raw from cleaned data array"""
        cleaned_raw = mne.io.RawArray(data, info)
        cleaned_raw.set_annotations(annotations)
        return cleaned_raw

    def _zapline_iter(
        self,
        fline: float = 50.0,
    ) -> BaseRaw:
        """apply zapline iterative method for line noise removal"""
        logger.info('Applying zapline iterative line noise removal...')
        raw = self.raw.copy()
        data = self._make_zapline_feature(raw)
        sfreq = raw.info['sfreq']
        with contextlib.redirect_stdout(io.StringIO()):  # mute for simplicity
            out, _ = dss.dss_line_iter(data, fline, sfreq=sfreq, nfft=400)
            data_clean = out.T.squeeze()

        return self._reconstruct_raw(
            data_clean,
            self.raw.annotations,
            self.raw.info,
        )

    def _zapline(
        self,
        fline: float = 50.0,
        nremove: int | None = None,
        removing_ratio: float = 0.22,
    ) -> BaseRaw:
        """apply zapline method for line noise removal"""
        logger.info('Applying zapline line noise removal...')
        raw = self.raw.copy()
        nremove = (
            int(len(raw.info['ch_names']) * removing_ratio)
            if nremove is None
            else nremove
        )
        data = self._make_zapline_feature(raw)
        sfreq = raw.info['sfreq']
        with contextlib.redirect_stdout(io.StringIO()):  # mute for simplicity
            out, _ = dss.dss_line(
                data,
                fline=fline,
                sfreq=sfreq,
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

    def plot_psd_comparison(self, raw, clean, title=''):
        with contextlib.redirect_stdout(io.StringIO()):  # mute for simplicity
            plt.close('all')
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            raw.plot_psd(ax=axes[0], show=False)
            clean.plot_psd(ax=axes[1], show=False)
            axes[0].set_title('Before')
            axes[1].set_title('After')
            fig.suptitle(title)
            plt.close(fig)
        return fig

    def run(
        self,
        fline: float = 50.0,
        find_in: list | None = None,
        save_deriv: bool = True,
        fname: str | None = None,
    ) -> BaseRaw:
        self.raw = self._pick_chs(self.raw, find_in)
        match self.dtype:
            case 'meg':
                clean = self._zapline(fline)
            case 'eeg':
                clean = self._zapline_iter(fline)
            case _:
                raise ValueError("datatype must be 'eeg' or 'meg'")

        if save_deriv:
            if fname is None:
                raise ValueError(
                    'Please provide a filename to save the derivative.',
                )
            fname = Path(f'{fname}_desc-psdplot_{self.dtype}.png')
            fig = self.plot_psd_comparison(
                self.raw, clean, title='Line Noise Removal Comparison',
            )
            fig.savefig(fname)
            logger.success(f'Saved PSD comparison plot to {fname}')

        return clean
