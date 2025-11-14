# A Human-in-the-loop M/EEG Preprocessing Pipeline
# %%
from __future__ import annotations

from typing import Optional
from typing import Union

import mne
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from mne_bids import BIDSPath
from mne_bids import read_raw_bids
from mne_bids import write_raw_bids

RANDOM_SEED = 42
# %%


class PrepPipeline:
    def __init__(
        self,
        bids: BIDSPath,
        preload: bool = True,
    ):
        self.bids = bids
        self.datatype = bids.datatype
        if preload:
            self.raw = read_raw_bids(
                bids, verbose=False,
            )

    def bad_chs_processing(self):
        ...
