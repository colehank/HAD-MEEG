# A Human-in-the-loop M/EEG Preprocessing Pipeline
# %%
from __future__ import annotations

from pathlib import Path

import mne
from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne_bids import BIDSPath
from mne_bids import read_raw_bids

from .bad_chs import BadChsRunner
from .ica import ICARunner
from .line_noise import LineNoiseRunner

RANDOM_SEED = 42


# %%
class PrepPipeline:
    def __init__(
        self,
        bids: BIDSPath,
        deri_root: str | Path | None = None,
        n_jobs: int = 1,
        use_cuda: bool = True,
        random_state: int = RANDOM_SEED,
    ):
        self.bids = bids
        self.bids_root = bids.root
        self.dtype = bids.datatype
        self.random_state = random_state

        if deri_root is None:
            self.prep_root = bids.root / "derivatives" / "preproc"
        else:
            self.prep_root = Path(deri_root) / "preproc"
        self.basename = bids.basename
        self.save_fname = (
            self.prep_root
            / f"sub-{bids.subject}"
            / f"ses-{bids.session}"
            / self.dtype
            / f"{self.basename}"
        )
        logger.trace(
            f"{self.dtype} Data derivatives will be saved to {self.save_fname}",
        )

        if use_cuda:
            logger.trace("Using CUDA for computations where possible.")
            self.n_jobs = "cuda"
            mne.cuda.init_cuda()
        else:
            self.n_jobs = n_jobs
            logger.trace(
                f"Using {n_jobs} CPU cores for computations where possible.",
            )

    def load(self) -> None:
        logger.info("Loading data...")
        self.raw = read_raw_bids(self.bids)
        self.raw.load_data()
        logger.success("Data loaded.")

    def preprep_raw(
        self,
        lowpass: float,
        highpass: float,
        sfreq: float,
    ) -> None:
        if not hasattr(self, "raw"):
            self.load()
        assert highpass < sfreq / 2, (
            "Highpass frequency must be lower than Nyquist frequency."
        )
        logger.info(
            f"Pre-preprocessing: {highpass=}Hz, {lowpass=}Hz, {sfreq=}Hz",
        )
        self.raw.filter(l_freq=highpass, h_freq=lowpass, n_jobs=self.n_jobs)
        self.raw.resample(sfreq, n_jobs=self.n_jobs)
        logger.success("Pre-preprocessing completed.")

    def run(
        self,
        preprep_params: dict | None = None,
        save: bool = True,
        regress: bool = False,
        manual_ica_checked: bool = False,
        manual_labels: list[str] | None = None,
        ica: ICA | None = None,
    ) -> BaseRaw | None:
        if regress and not manual_ica_checked:
            logger.error(
                "Regressing artifacts without manual checking."
                "which is not compatible in the current pipeline.",
            )
        if not save and regress:
            logger.error(
                "Saving must be enabled when regressing artifacts.",
            )
            return None

        if preprep_params is None:
            logger.trace("Using default preprep parameters.")
            lowpass = 100.0
            highpass = 0.1
            sfreq = 250.0
        else:
            lowpass = preprep_params.get("lowpass", 100.0)
            highpass = preprep_params.get("highpass", 0.1)
            sfreq = preprep_params.get("sfreq", 250.0)

        self.preprep_raw(
            lowpass=lowpass,
            highpass=highpass,
            sfreq=sfreq,
        )
        if manual_ica_checked:
            if ica is None or manual_labels is None:
                logger.trace(
                    "Manual ICA checked is True, but no ICA or manual labels provided."
                    "Trying find them from BIDS.",
                )
                fn_ica = self.save_fname.with_name(
                    self.save_fname.stem + f"_desc-ica_{self.dtype}.fif",
                )
                fn_labels = self.save_fname.with_name(
                    self.save_fname.stem + f"_desc-ica_{self.dtype}.tsv",
                )
                if fn_ica.is_file() and fn_labels.is_file():
                    ica = mne.preprocessing.read_ica(fn_ica)
                    import pandas as pd

                    df = pd.read_csv(fn_labels, sep="\t")
                    manual_labels = df["ic_type"].values.tolist()
                    logger.trace(
                        f"Loaded ICA and manual labels from {fn_ica} and {fn_labels}.",
                    )
                else:
                    raise ValueError(
                        "Cannot find ICA or manual labels from BIDS."
                        "Please provide them directly.",
                    )

        save_fname = self.save_fname
        # 1. Bad channel detection
        runner = BadChsRunner(raw=self.raw, random_state=self.random_state)
        self.raw = runner.run(fname=save_fname)
        logger.success("Bad channel detection completed.")

        # 2. Line noise removal
        runner = LineNoiseRunner(raw=self.raw, random_state=self.random_state)
        self.raw = runner.run(fname=save_fname)
        logger.success("Line noise removal completed.")

        # 3. Denoise
        if self.dtype == "eeg":
            logger.info("apply average reference to denosing")
            self.raw.set_eeg_reference("average")
        elif self.dtype == "meg":
            logger.info("apply gradient compensation 3 to denosing")
            self.raw.apply_gradient_compensation(3)
        logger.success("Reference denoising completed.")

        # 4. ICA artifact removal
        runner = ICARunner(raw=self.raw, random_state=self.random_state)
        self.raw = runner.run(
            regress=regress,
            fname=save_fname,
            manual=manual_ica_checked,
            manual_labels=manual_labels,
            ica=ica,
        )
        logger.success(
            f"ICA ({'automatic' if not manual_ica_checked else 'manual'} label"
            f"{' -> regression' if regress else ''}) completed.",
        )

        # (5). EEG only: set average reference again after ICA
        if self.dtype == "eeg" and regress:
            logger.info("re-referencing EEG after ICA.")
            self.raw.set_eeg_reference("average")
            logger.success("re-reference EEG done.")

        if save:
            logger.info("Saving final clean BaseRaw.")
            save_fname = self.save_fname.with_name(
                self.save_fname.stem + f"_preproc_{self.dtype}.fif",
            )
            self.raw.save(save_fname, overwrite=True)
            logger.trace(f"Final clean BaseRaw saved to {save_fname}")
            logger.success("Cleaned data saved")

        logger.success("Preprocessed!")
