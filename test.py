# %%
from __future__ import annotations

from src.config import DataConfig
# from src.prep import PrepPipeline
from src.prep import BatchPrepPipeline

from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("batch.log", level="TRACE")
# sub = '01'
# all_bids = cfg.source[sub]

# # %%
# meg_bids = cfg.source[sub]['meg'][0]
# eeg_bids = cfg.source[sub]['eeg'][0]
# # meg_pipe = PrepPipeline(bids=meg_bids)
# eeg_pipe = PrepPipeline(bids=eeg_bids)
# # %%
# # meg_pipe.run(save=False)
# eeg_pipe.run(save=False)
# # %%
# pip_1 = PrepPipeline(bids=eeg_bids)
# clean = pip_1.run(
#     save=True,
#     manual_ica_checked = True,
#     regress=True,
# )
# %%
if __name__ == '__main__':
    cfg = DataConfig()

    all_bids = cfg.source_bids_list
    batch_pipe = BatchPrepPipeline(
        bids_list=all_bids,
        use_cuda=True,
    )
    batch_pipe.run(
        manual_ica_checked=False,
        regress=False,
    )
# %%
