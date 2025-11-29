# %%
import mne
from mne.io import BaseRaw, read_raw
from loguru import logger
from tqdm.auto import tqdm
import pandas as pd
from mne_bids import get_entities_from_fname as parse_fname
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


mne.cuda.init_cuda()


class Epocher:
    def __init__(self, config) -> None:
        """Epoching class to handle epoching of preprocessed data.

        Parameters
        ----------
        config : src.DataConfig
            Configuration object containing data paths and settings.
        """
        df = config.source_df

        grouped = df.groupby(["subject", "datatype"])["preprocessed"].apply(list)

        preped_fps: dict = {}
        for (sub, dtype), fps in grouped.items():
            preped_fps.setdefault(sub, {})[dtype] = fps

        self.preped_fps = preped_fps
        self.subs_dtypes = {
            sub: list(dtypes.keys()) for sub, dtypes in preped_fps.items()
        }
        self.events_fps = config.detailed_events

        logger.info(f"{len(self.subs_dtypes)} subjects found")
        lines = []
        for sub, dtypes in preped_fps.items():
            meg_n = len(dtypes.get("meg", []))
            eeg_n = len(dtypes.get("eeg", []))
            lines.append(f"\n sub-{sub} has {meg_n} meg raws, {eeg_n} eeg raws")
        logger.trace("\n".join(lines))

        self.save_dir = config.derivatives_root / "epochs"
        self.save_dir.mkdir(exist_ok=True)

    def run_sub(
        self,
        sub: str,
        tmin: float = -0.1,
        tmax: float = 2,
        stimid: str = "video on",
        baseline: tuple[float, float] = (None, 0.0),
        baseline_mode: str = "zscore",
        highpass: float = 0.1,
        lowpass: float = 40.0,
        sfreq: float = 200.0,
        save: bool = True,
    ) -> dict[str, mne.Epochs]:
        """Run epoching for a single subject."""
        sub_epo: dict[str, list[mne.Epochs]] = {}
        metadata = pd.read_csv(self.events_fps[sub])
        preped_raws = self._prep_sub(
            sub, highpass=highpass, lowpass=lowpass, sfreq=sfreq
        )
        for dtype, raws in preped_raws.items():
            epos = self._raws_to_epos(
                raws=raws,
                metadata=metadata,
                tmin=tmin,
                tmax=tmax,
                stimid=stimid,
                baseline=baseline,
                baseline_mode=baseline_mode,
            )
            epo = mne.concatenate_epochs(epos, add_offset=True)
            if save:
                fname = self.save_dir / f"sub-{sub}_epo_{dtype}.fif"
                epo.save(fname, overwrite=True)
            sub_epo[dtype] = epo

        return sub_epo

    def run(
        self,
        n_jobs: int = 1,
        tmin: float = -0.1,
        tmax: float = 2,
        stimid: str = "video on",
        baseline: tuple[float, float] = (None, 0.0),
        baseline_mode: str = "zscore",
        highpass: float = 0.1,
        lowpass: float = 40.0,
        sfreq: float = 200.0,
    ) -> None:
        """Run epoching for all subjects in parallel using joblib.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs for joblib.Parallel.
            Use -1 to use all available cores.
        tmin, tmax, stimid, baseline, baseline_mode :
            Passed directly to `run_sub`.
        """
        subs = sorted(self.subs_dtypes.keys())
        logger.info(
            f"Start epoching {len(subs)} subjects with n_jobs={n_jobs}, "
            f"tmin={tmin}, tmax={tmax}, stimid={stimid}, "
            f"baseline={baseline}, baseline_mode={baseline_mode}"
        )

        with tqdm_joblib(
            total=len(subs),
            desc="Epoching subjects",
            leave=True,
            position=0,
        ):
            _ = Parallel(n_jobs=n_jobs)(
                delayed(self.run_sub)(
                    sub=sub,
                    tmin=tmin,
                    tmax=tmax,
                    stimid=stimid,
                    baseline=baseline,
                    baseline_mode=baseline_mode,
                    highpass=highpass,
                    lowpass=lowpass,
                    sfreq=sfreq,
                )
                for sub in subs
            )
            del _
        logger.info("All subjects epoching finished.")

    def _raws_to_epos(
        self,
        raws: list[BaseRaw],
        metadata: pd.DataFrame,
        tmin: float = -0.1,
        tmax: float = 2,
        stimid: str = "video on",
        align_stimcode=1,
        baseline: tuple[float, float] = (None, 0.0),
        baseline_mode: str = "zscore",
    ) -> list[mne.Epochs]:
        """Convert list of raws to epochs based on events."""
        epos: list[mne.Epochs] = []
        for raw in raws:
            entities = parse_fname(raw.filenames[0])
            sub = entities["subject"]
            ses = entities["session"]
            task = entities["task"]
            run = entities["run"]

            meta = metadata[
                (metadata["subject"] == int(sub))
                & (metadata["session"] == ses)
                & (metadata["task"] == task)
                & (metadata["run"] == int(run))
            ]

            # ev, evid = mne.events_from_annotations(raw)
            ev, _ = mne.events_from_annotations(raw, event_id={stimid: align_stimcode})
            filtered_ev = ev

            epo = mne.Epochs(
                raw=raw,
                events=filtered_ev,
                event_id={stimid: align_stimcode},
                tmin=tmin,
                tmax=tmax,
                metadata=meta,
                baseline=baseline,
            )
            epo = self._baseline_correction(
                epo,
                baseline=baseline,
                mode=baseline_mode,
            )
            epos.append(epo)
        return epos

    def _load_raw(
        self,
        fp: str,
    ) -> BaseRaw:
        """Load preprocessed raw data from file path."""
        raw = read_raw(fp, preload=True)
        return raw

    def _prep_raw(
        self,
        raw: BaseRaw,
        dtype: str,
        highpass: float = 0.1,
        lowpass: float = 40.0,
        sfreq: float = 200.0,
    ) -> BaseRaw:
        """Apply filtering and resampling to raw data."""
        if dtype == "eeg":
            raw.set_eeg_reference("average", projection=False)
            raw.set_channel_types(
                {ch: "misc" for ch in ["M1", "M2"] if ch in raw.ch_names}
            )

            raw.pick_types(eeg=True)
        elif dtype == "meg":
            raw.pick_types(meg=True)

        raw.resample(sfreq=sfreq)
        raw.filter(l_freq=highpass, h_freq=lowpass)
        return raw

    def _prep_sub(
        self,
        sub: str,
        highpass: float = 0.1,
        lowpass: float = 40.0,
        sfreq: float = 200.0,
    ) -> dict[str, list[BaseRaw]]:
        """Prep one sub's raw data."""
        fps = {dtype: self.preped_fps[sub][dtype] for dtype in self.subs_dtypes[sub]}
        preped_raws: dict[str, list[BaseRaw]] = {}

        for dtype, fp_list in fps.items():
            raws = [
                self._load_raw(fp)
                for fp in tqdm(
                    fp_list, desc=f"Loading {dtype} raws", leave=False, position=1
                )
            ]
            raws = [
                self._prep_raw(raw, dtype, highpass, lowpass, sfreq)
                for raw in tqdm(
                    raws,
                    desc=f"Prep {dtype} raws: ({highpass}-{lowpass}Hz, {sfreq}Hz)",
                    leave=False,
                    position=1,
                )
            ]
            if dtype == "meg":
                raws = self._align_meg_head(raws)
            preped_raws[dtype] = raws

        return preped_raws

    def _align_meg_head(
        self,
        raws: list[BaseRaw],
        ref_idx: int = 0,
    ) -> list[BaseRaw]:
        """Align CTF MEG data to head position using specified method."""
        logger.trace(f"{len(raws)} CTF raws to be aligned, ref: the first raw")
        ref_info = raws[ref_idx].info
        for raw in tqdm(
            raws[1:], desc="Aligning MEG head positions", position=1, leave=False
        ):
            this_info = raw.info
            map = mne.forward._map_meg_or_eeg_channels(
                ref_info, this_info, mode="accurate", origin=(0.0, 0.0, 0.04)
            )
            ori_data = raw.get_data(picks="meg")
            aligned_data = map @ ori_data

            raw._data = aligned_data
            raw.info.update(
                {
                    "dev_head_t": ref_info["dev_head_t"],
                }
            )
        return raws

    def _baseline_correction(
        self,
        epochs: mne.Epochs,
        baseline: tuple[float, float] = (None, 0.0),
        mode: str = "zscore",
    ) -> mne.Epochs:
        """Apply baseline correction to epochs."""
        baselined = mne.baseline.rescale(
            data=epochs.get_data(), times=epochs.times, baseline=baseline, mode=mode
        )
        epochs = mne.EpochsArray(
            baselined,
            info=epochs.info,
            events=epochs.events,
            tmin=epochs.tmin,
            event_id=epochs.event_id,
            metadata=epochs.metadata,
        )
        return epochs
