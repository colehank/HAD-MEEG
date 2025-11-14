# %%
from __future__ import annotations

from mne_bids import read_raw_bids

from src import DataConfig
from src.prep.bad_chs import BadChsRunner
from src.prep.line_noise import LineNoiseRunner
# %%
config = DataConfig()
test_meg = read_raw_bids(config.source['01']['meg'][0])
test_eeg = read_raw_bids(config.source['01']['eeg'][0])


def make_test(raw):
    raw.crop(60)
    raw.resample(250)
    return raw


toy_meg = make_test(test_meg)
toy_eeg = make_test(test_eeg)
# %%
lr = LineNoiseRunner(toy_meg)
# %%
