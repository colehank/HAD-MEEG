from __future__ import annotations

from typing import Tuple

from src import config, Epocher


# Runtime configuration
N_JOBS: int = 8
TMIN: float = -0.1
TMAX: float = 2.0
STIM_ID: str = "video on"
BASELINE: Tuple[float | None, float] = (None, 0.0)
BASELINE_MODE: str = "zscore"
HIGHPASS: float = 0.1
LOWPASS: float = 40.0
SFREQ: float = 200.0


def main() -> None:
    cfg = config.DataConfig()
    epocher = Epocher(cfg)
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


if __name__ == "__main__":
    main()
