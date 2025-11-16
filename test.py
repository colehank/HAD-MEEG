# %%
from __future__ import annotations

from mne_bids import BIDSPath

from src.prep import PrepPipeline
# %%


def get_test_bids() -> BIDSPath:
    ROOT = 'resources/toy_bids'
    test_bids = BIDSPath(
        subject='01',
        session='01',
        datatype='meg',
        task='rest',
        root=ROOT,
    )
    test_bids_eeg = test_bids.copy().update(datatype='eeg')
    return {'meg': test_bids, 'eeg': test_bids_eeg}


meg_bdis = get_test_bids()['meg']
eeg_bdis = get_test_bids()['eeg']
meg_pipe = PrepPipeline(bids=meg_bdis)
eeg_pipe = PrepPipeline(bids=eeg_bdis)
# %%
meg_pipe.run(save=False)
eeg_pipe.run(save=False)
# %%
