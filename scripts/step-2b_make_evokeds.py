# %%
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import DataConfig, AnalyseConfig
from src.evo import EpoToEvo
from src.epo import concat_epochs
from joblib import Parallel, delayed, dump
from tqdm_joblib import tqdm_joblib
from tqdm.auto import tqdm
import mne
from loguru import logger

cfg_data = DataConfig()
cfg_anal = AnalyseConfig()
N_JOBS = cfg_anal.n_jobs
SAVE_DIR = cfg_data.results_root / "evos"
GRAND_SAVE_DIR = SAVE_DIR / "grand_evo"
SUB_SAVE_DIR = SAVE_DIR / "sub_evo"
GRAND_SAVE_DIR.mkdir(parents=True, exist_ok=True)
SUB_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# %%
def read_epo(fp: str) -> mne.Epochs:
    epo = mne.read_epochs(fp, verbose="ERROR")
    return epo


def read_epos(fp_list: list[str], n_jobs: int = 10) -> list[mne.Epochs]:
    with tqdm_joblib(
        total=len(fp_list),
        desc="Loading epochs",
    ):
        epos = Parallel(n_jobs=n_jobs)(delayed(read_epo)(fp) for fp in fp_list)
    return epos


def concat_epos(
    epos: list[mne.Epochs],
    align_head: bool = True,
    n_jobs: int = 10,
) -> mne.Epochs:
    epo = concat_epochs(epos, align_head=align_head, n_jobs=n_jobs)
    return epo


def avg_evo_by_meta(
    epo: mne.Epochs,
    by_col: str,
    remaining_cols: list[str] | None = None,
    n_jobs: int = 10,
) -> EpoToEvo:
    evo_set = EpoToEvo(
        epo,
        by_col=by_col,
        remaining_cols=remaining_cols,
        n_jobs=n_jobs,
    )
    return evo_set


# %%
if __name__ == "__main__":
    fp_meg_epos = (
        cfg_data.source_df.query("datatype == 'meg'")["epochs"].unique().tolist()
    )
    fp_eeg_epos = (
        cfg_data.source_df.query("datatype == 'eeg'")["epochs"].unique().tolist()
    )
    # %%
    logger.info("Reading, concatenating and averaging epochs ")

    eeg_epos = read_epos(fp_eeg_epos, n_jobs=10)
    eeg_epo = concat_epos(eeg_epos, align_head=False, n_jobs=10)
    eeg_evo = avg_evo_by_meta(
        eeg_epo,
        by_col="class_id",
        remaining_cols=[
            "class_name",
            "raw_superclass_name",
            "superclass_level0",
            "superclass_level1",
            "superclass_level2",
        ],
        n_jobs=10,
    )
    dump(eeg_evo, GRAND_SAVE_DIR / "grand_evo_eeg.pkl")
    del eeg_epos
    del eeg_epo
    del eeg_evo

    meg_epos = read_epos(fp_meg_epos, n_jobs=10)
    meg_epo = concat_epos(meg_epos, align_head=True, n_jobs=10)
    meg_evo = avg_evo_by_meta(
        meg_epo,
        by_col="class_id",
        remaining_cols=[
            "class_name",
            "raw_superclass_name",
            "superclass_level0",
            "superclass_level1",
            "superclass_level2",
        ],
        n_jobs=10,
    )
    dump(meg_evo, GRAND_SAVE_DIR / "grand_evo_meg.pkl")
    del meg_epos
    del meg_epo
    del meg_evo
    # %%
    logger.info("Processing subject-wise evoked data.")
    for fp in tqdm(fp_eeg_epos + fp_meg_epos):
        fp = Path(fp)
        fn = fp.name
        save_fp = SUB_SAVE_DIR / fn.replace("epo", "evo").replace(".fif", ".pkl")
        epo = mne.read_epochs(fp)
        evo = avg_evo_by_meta(
            epo,
            by_col="class_id",
            remaining_cols=[
                "subject",
                "session",
                "class_name",
                "raw_superclass_name",
                "superclass_level0",
                "superclass_level1",
                "superclass_level2",
            ],
            n_jobs=30,
        )
        dump(evo, save_fp)
# %%
