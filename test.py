# %%
from __future__ import annotations
from src.config import DataConfig
from src.prep import BatchPrepPipeline
# %%
if __name__ == '__main__':
    cfg = DataConfig()
    all_bids = cfg.source_bids_list
    batch_pipe = BatchPrepPipeline(
        bids_list=all_bids,
        use_cuda=True,
        n_jobs=8,
        random_state=42,
    )
    batch_pipe.run(
        manual_ica_checked=False,
        regress=False,
    )