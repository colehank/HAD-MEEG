# %%
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Tuple
from src import Epocher, DataConfig, AnalyseConfig

cfg_data = DataConfig()
cfg_anal = AnalyseConfig()
epocher = Epocher(cfg_data)
# %%
# Runtime configuration
N_JOBS: int = cfg_anal.n_jobs
TMIN: float = -0.1  # epochs' start time
TMAX: float = 2.0  # epochs' end time
STIM_ID: str = "video on"  # event to epoch around
BASELINE: Tuple[float | None, float] = (None, 0.0)  # baseline correction period
BASELINE_MODE: str = "zscore"  # baseline correction mode
HIGHPASS: float = 0.1  # highpass filter cutoff
LOWPASS: float = 40.0  # lowpass filter cutoff
SFREQ: float = 250.0  # resampling frequency


if __name__ == "__main__":
    epocher.run(
        n_jobs=N_JOBS,
        tmin=TMIN,
        tmax=TMAX,
        stimid=STIM_ID,
        baseline=BASELINE,
        baseline_mode=BASELINE_MODE,
        highpass=HIGHPASS,
        lowpass=LOWPASS,
        sfreq=SFREQ,
    )
