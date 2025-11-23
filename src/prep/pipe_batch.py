from __future__ import annotations

from pathlib import Path

from joblib import delayed
from joblib import Parallel
from mne_bids import BIDSPath
from tqdm_joblib import tqdm_joblib

from ..logger import setup_logging
from .pipe_single import PrepPipe

RANDOM_STATE = 42


class BatchPrepPipe:
    def __init__(
        self,
        bids_list: list[BIDSPath],
        deri_root: str | Path | None = None,
        use_cuda: bool = True,
        random_state: int = RANDOM_STATE,
        n_jobs: int = 1,
    ):
        self.bids_list = bids_list
        self.deri_root = deri_root
        self.use_cuda = use_cuda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _run_one(
        self,
        bids: BIDSPath,
        manual_ica_checked: bool,
        manual_labels: dict[str, list[str]] | None = None,
        regress: bool = False,
        logging_level: str = "WARNING",
        logging_fname: str | None = None,
        skip_raw: bool = True,
        skip_ica: bool = True,
    ) -> None:
        if self.deri_root is None:
            deri_root = bids.root / "derivatives"
        else:
            deri_root = self.deri_root
        deriv_raw = (
            Path(deri_root)
            / "preproc"
            / f"sub-{bids.subject}"
            / f"ses-{bids.session}"
            / bids.datatype
            / f"{bids.basename}_preproc_{bids.datatype}.fif"
        )
        deriv_ica = (
            Path(deri_root)
            / "preproc"
            / f"sub-{bids.subject}"
            / f"ses-{bids.session}"
            / bids.datatype
            / f"{bids.basename}_desc-ica_{bids.datatype}.fif"
        )
        if skip_raw and deriv_raw.exists():
            return
        if skip_ica and deriv_ica.exists():
            return
        if logging_fname is None:
            sub = f"prep_sub-{bids.subject}"
            ses = f"ses-{bids.session}"
            dtype = bids.datatype
            subdir = Path(sub) / Path(ses) / Path(dtype)
            logging_fname = str(subdir / f"{bids.basename}_desc-prepclog_{dtype}")
        setup_logging(
            stdout_level=logging_level,
            parallel=self.n_jobs != 1,
            fname=logging_fname,
        )
        try:
            pipe = PrepPipe(
                bids=bids,
                deri_root=self.deri_root,
                use_cuda=self.use_cuda,
                random_state=self.random_state,
            )
            pipe.run(
                save=manual_ica_checked,
                manual_ica_checked=manual_ica_checked,
                manual_labels=manual_labels[bids.basename]["_manual_labels"]
                if manual_labels
                else None,
                regress=regress,
            )
        except Exception as e:
            from loguru import logger

            logger.error(f"Error processing {bids.basename}: {e}")

    def run(
        self,
        manual_ica_checked: bool = False,
        manual_labels: dict[str, list[str]] | None = None,
        regress: bool = False,
        logging_level: str = "WARNING",
        logging_fname: str = None,
        skip_raw: bool = True,
        skip_ica: bool = True,
    ) -> None:
        with tqdm_joblib(
            total=len(self.bids_list),
            desc=f"Cooking in {self.n_jobs} jobs",
        ):
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_one)(
                    bids,
                    manual_ica_checked=manual_ica_checked,
                    manual_labels=manual_labels,
                    regress=regress,
                    skip_raw=skip_raw,
                    skip_ica=skip_ica,
                )
                for bids in self.bids_list
            )
