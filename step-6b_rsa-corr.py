# %%
import numpy as np
from pathlib import Path
from joblib import load, dump
from src.utils import get_soi_picks
from src.evo import EvokedSet
from src.rsa import TimeRDM, RSA
from loguru import logger

# import matplotlib.pyplot as plt
# %%
EVO_DIR = Path("../HAD-MEEG_results/grand_evo")
RES_DIR = Path("../HAD-MEEG_results/rsa")
SOIS = ["O", "P", "C", "F", "T", "OT", "all"]  # (M/EEG) sensor of interests
ROIS = ["EV", "VS", "DS", "LS"]  # (fMRI) regions of interest
ROI_MAP = {"EV": "Early", "VS": "Ventral", "DS": "Dorsal", "LS": "Lateral"}
N_ITER = 1000  # number of iterations for bootstrap
ALPHA = 0.05  # significance level for confidence iterval

RES_DIR.mkdir(parents=True, exist_ok=True)


# %%
def get_class_feature(
    evo: EvokedSet,
    order: np.ndarray,
    soi: str,
):
    """Get the M/EEG's sensor data of
    the specified SOI and class order.

    Parameters
    ----------
    evo : EvokedSet
        The evoked set.
    order : np.ndarray
        The order of the classes in 1d array,
        each element is the class_name(str).
    soi : str
        The SOI name,
        e.g., 'O', 'P', 'C', 'F', 'T', 'OT'.
    """
    picks = get_soi_picks(evo[0], soi)
    data = []
    for ori in order:
        this_evo = evo[evo.metadata["class_name"] == ori].copy()
        this = this_evo.pick(picks).data  # nchan x ntime
        data.append(this)
    return np.array(data)  # nclass x nchan x ntime


def fusion(
    space_rdm: np.ndarray,
    time_rdms: np.ndarray,
    corr_method: str = "spearman",
    sig_method: str = "bootstrap",
    n_jobs: int = 1,
    n_iter: int = 1000,
    alpha: float = 0.05,
    input_type: str = "one2n",
):
    rsa = RSA(
        rdm1=space_rdm,
        rdm2=time_rdms,
        input_type=input_type,
        n_jobs=n_jobs,
        n_iter=n_iter,
        alpha=alpha,
    )
    corr, sig = rsa.compute(
        corr_method=corr_method,
        sig_method=sig_method,
    )
    return corr, sig


# %%
if __name__ == "__main__":
    mevo: EvokedSet = load(EVO_DIR / "grand_evo_meg.pkl")
    eevo: EvokedSet = load(EVO_DIR / "grand_evo_eeg.pkl")
    fmri_rdms = np.load("resources/fmri-rdms.npz")

    classorder = fmri_rdms["order"]

    results = {}
    for roi in ROIS:
        results[roi] = {}
        rdm = fmri_rdms[roi]  # nclass x nclass
        for soi in SOIS:
            logger.info(f"Processing ROI: {roi}, SOI: {soi}")
            meg_feat = get_class_feature(
                mevo, classorder, soi=soi
            )  # nclass x nchan x ntime
            eeg_feat = get_class_feature(
                eevo, classorder, soi=soi
            )  # nclass x nchan x ntime
            meg_rdms = TimeRDM.compute(
                meg_feat, metric="correlation", normalize=True
            )  # ntime x nclass x nclass
            eeg_rdms = TimeRDM.compute(
                eeg_feat, metric="correlation", normalize=True
            )  # ntime x nclass x nclass
            meg_corr, meg_sig = fusion(
                space_rdm=rdm,
                time_rdms=meg_rdms,
                n_jobs=10,
                n_iter=N_ITER,
                alpha=ALPHA,
            )
            eeg_corr, eeg_sig = fusion(
                space_rdm=rdm,
                time_rdms=eeg_rdms,
                n_jobs=10,
                n_iter=N_ITER,
                alpha=ALPHA,
            )
            results[roi][soi] = {
                "meg_corr": meg_corr,
                "meg_sig": meg_sig,
                "eeg_corr": eeg_corr,
                "eeg_sig": eeg_sig,
            }
    dump(results, RES_DIR / "results.pkl")
    # %%
    results = load(RES_DIR / "results.pkl")
    for soi in SOIS:
        logger.info(f"Processing MEG-vs-EEG fusion for SOI: {soi}")
        meg_feat = get_class_feature(
            mevo, classorder, soi=soi
        )  # nclass x nchan x ntime
        eeg_feat = get_class_feature(
            eevo, classorder, soi=soi
        )  # nclass x nchan x ntime
        meg_rdms = TimeRDM.compute(
            meg_feat, metric="correlation", normalize=True
        )  # ntime x nclass x nclass
        eeg_rdms = TimeRDM.compute(
            eeg_feat, metric="correlation", normalize=True
        )  # ntime x nclass x nclass
        meg_eeg_corr, meg_eeg_sig = fusion(
            space_rdm=meg_rdms,
            time_rdms=eeg_rdms,
            input_type="n2n",
            n_jobs=10,
            n_iter=N_ITER,
            alpha=ALPHA,
        )
        results["MEG vs. EEG"] = {
            "meg_eeg_corr": meg_eeg_corr,
            "meg_eeg_sig": meg_eeg_sig,
        }
    dump(results, RES_DIR / "results.pkl")
# %%
