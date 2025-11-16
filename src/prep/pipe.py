# A Human-in-the-loop M/EEG Preprocessing Pipeline
# %%
from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Union

from loguru import logger
from mne.io import BaseRaw
from mne_bids import BIDSPath
from mne_bids import read_raw_bids

from .bad_chs import BadChsRunner
from .ica import ICARunner
from .line_noise import LineNoiseRunner
RANDOM_SEED = 42
# %%


class PrepPipeline:
    def __init__(
        self,
        bids: BIDSPath,
        deri_root: str | Path | None = None,
    ):
        self.bids = bids
        self.bids_root = bids.root
        self.dtype = bids.datatype

        if deri_root is None:
            self.prep_root = bids.root / 'derivatives' / 'preprocessing'
        else:
            self.prep_root = Path(deri_root) / 'preprocessing'
        self.basename = bids.basename
        self.save_fname = self.prep_root / \
            f'sub-{bids.subject}' / f'ses-{bids.session}' / \
            self.dtype / f'{self.basename}'
        logger.info(f'Derivatives will be saved to {self.save_fname}')

    def load(self) -> None:
        self.raw = read_raw_bids(self.bids)

    def run(self, save: bool = True) -> BaseRaw:
        self.load()
        save_fname = self.save_fname
        # 1. Bad channel detection
        runner = BadChsRunner(raw=self.raw)
        self.raw = runner.run(fname=save_fname)

        # 2. Line noise removal
        runner = LineNoiseRunner(raw=self.raw)
        self.raw = runner.run(fname=save_fname)

        # 3. Denoise
        if self.dtype == 'eeg':
            self.raw.set_eeg_reference('average')
        elif self.dtype == 'meg':
            self.raw.apply_gradient_compensation(3)

        # 4. ICA artifact removal
        runner = ICARunner(raw=self.raw)
        self.raw = runner.run(fname=save_fname)

        # (5). EEG only: set average reference again after ICA
        if self.dtype == 'eeg':
            self.raw.set_eeg_reference('average')

        if save:
            save_fname = self.save_fname.with_name(
                self.save_fname.stem + '_desc-preproc.fif',
            )
            self.raw.save(save_fname, overwrite=True)
            logger.success(f'Saved preprocessed data to {save_fname}')
