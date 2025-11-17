from __future__ import annotations

from pathlib import Path

from mne_bids import BIDSPath
from tqdm.auto import tqdm

from .pipe_single import PrepPipeline


class BatchPrepPipeline:
    def __init__(
        self,
        bids_list: list[BIDSPath],
        deri_root: str | Path | None = None,
        use_cuda: bool = True,
    ):
        self.bids_list = bids_list
        self.deri_root = deri_root
        self.use_cuda = use_cuda

    def run(
        self,
        manual_ica_checked: bool = False,
        manual_labels: list[str] | None = None,
        regress: bool = False,
    ) -> None:
        for bids in tqdm(self.bids_list, desc='Processing BIDS datasets'):
            pipe = PrepPipeline(
                bids=bids,
                deri_root=self.deri_root,
                use_cuda=self.use_cuda,
            )
            pipe.run(
                save=True,
                manual_ica_checked=manual_ica_checked,
                manual_labels=manual_labels,
                regress=regress,
            )
