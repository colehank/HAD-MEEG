# A Human-in-the-loop M/EEG Preprocessing Pipeline

# %%
import mne
from mne.io import BaseRaw
from mne_bids import (
    BIDSPath, 
    read_raw_bids, 
    write_raw_bids
)

import numpy as np
import pandas as pd
from typing import Optional, Union

RANDOM_SEED = 42
# %%
class PrepPipeline:
    def __init__(
        self, 
        bids: BIDSPath,
        preload: bool = True
        ):
        self.bids = bids
        self.datatype = bids.datatype
        if preload:
            self.raw = read_raw_bids(
                bids, verbose=False)
        
    def bad_chs_processing(self):
        ...
