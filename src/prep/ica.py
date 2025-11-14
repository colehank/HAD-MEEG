# %%
from mne.io import BaseRaw
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels
from typing import Optional
from .bids_loader import BaseLoader

class BadChsRunner(BaseLoader):
    def __init__(
        self,
        raw: BaseRaw,
        dtype: Optional[str] = None,
        random_state: int = 42,
        ):
        super().__init__(raw, dtype, random_state)
    
    def _feature_extra():
        ...