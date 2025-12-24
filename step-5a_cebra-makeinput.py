import numpy as np
from joblib import load, dump
from src.utils import get_soi_picks
from src.evo import EvokedSet
from src import DataConfig
import mne
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

cfg = DataConfig()
EVO_DIR = cfg.results_root / "evos" / "grand_evo"
SAVE_DIR = cfg.results_root / "cebra"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FONT_SIZE = 12
FONT_PATH = Path("resources") / "Helvetica.ttc"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()


def load_evo(dtype: str) -> EvokedSet:
    """Load the evoked set for the specified datatype."""
    if dtype == "meg":
        evo = load(EVO_DIR / "grand_evo_meg.pkl")
    elif dtype == "eeg":
        evo = load(EVO_DIR / "grand_evo_eeg.pkl")
    return evo


def get_times(evo: EvokedSet) -> np.ndarray:
    """Get the time points from the evoked set."""
    return evo[0].times


def get_cls_names(
    evo: EvokedSet,
    by: str = "class_name",
    unique: bool = True,
) -> np.ndarray:
    """Get the class names from the evoked set metadata."""
    if unique:
        class_names = evo.metadata[by].unique().tolist()
    else:
        class_names = evo.metadata[by].tolist()
    return np.array(class_names)


def get_cls_data(
    evo: EvokedSet,
    by: str,
    soi: str,
    avg_multiple: bool = False,
) -> dict[str, np.ndarray]:
    """Get the data for each class in the specified SOI."""
    picks = get_soi_picks(evo[0], soi)
    cls_names = get_cls_names(evo, by=by)
    cls_data = {}
    for cls_ in cls_names:
        this_evo = evo[evo.metadata[by] == cls_].copy()
        match this_evo:
            case list() if avg_multiple:
                this_evo = mne.grand_average(this_evo)
                this_data = this_evo.pick(picks).data
            case list() if not avg_multiple:
                this_data = np.array([ev.pick(picks).data for ev in this_evo])
            case mne.Evoked():
                this_data = this_evo.pick(picks).data
            case _:
                raise TypeError(f"Unsupported evoked type: {type(this_evo)}")

        cls_data[cls_] = this_data
    return cls_data


def report_cls_data(cls_data: dict[str, np.ndarray]):
    """Report the shape of data for each class."""
    report = {}
    for cls_, data in cls_data.items():
        report[cls_] = data.shape
    return report


def balance_cls_data(
    cls_data: dict[str, np.ndarray],
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Balance the number of samples for each class data."""
    report = report_cls_data(cls_data)
    n_samples = min(shape[0] for shape in report.values())
    rng = np.random.default_rng(random_state)
    balanced_data = {}
    for cls_, data in cls_data.items():
        n_current = data.shape[0]
        if n_current >= n_samples:
            selected_indices = rng.choice(n_current, size=n_samples, replace=False)
        else:
            selected_indices = rng.choice(n_current, size=n_samples, replace=True)
        balanced_data[cls_] = data[selected_indices]
    return balanced_data


def make_trial_label(
    cls_label: np.int_,
    non_cls_label: np.int_,
    trial: np.ndarray,  # n_channels x n_times
    times: np.ndarray,  # 1D array of time points, same length as trial's time
) -> np.ndarray:  # n_times x 2: [time, label]
    """Make trial labels for each time point in a trial."""
    assert times.shape[0] == trial.shape[1], (
        f"Time dimension mismatch: times={times.shape[0]}, trial={trial.shape[1]}"
    )

    cls_labels = np.where(times <= 0, non_cls_label, cls_label)
    labels = np.vstack((times, cls_labels)).T  # (n_times, 2): [time, class_id]
    return labels


def make_input(
    cls_data: dict[str, np.ndarray], times: np.ndarray, baseline_label="*Pre-stimulus"
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 per-trial 的 MEG 数据展开成 CEBRA 需要的 2D X, Y。

    输入
    ----
    cls_data: dict[class_name -> data]
        data 形状通常是 (n_trials, n_channels, n_times)
    times: np.ndarray
        形状: (n_times,)，通常来自 evo[0].times

    输出
    ----
    X: np.ndarray
        (N_total_samples, n_channels)
    Y: np.ndarray
        (N_total_samples, 2)，每行 [time, class_id]
    """
    clss = list(cls_data.keys())

    # 建立 class -> int 的映射
    # le = LabelEncoder()
    # le.fit(clss)
    n_cls = len(clss)
    cls_label = {c: np.int_(n_cls - idx) for idx, c in enumerate(clss)}
    cls_label[baseline_label] = np.int_(0)

    X_list = []
    Y_list = []

    for cls_, data in cls_data.items():
        data = np.asarray(data)

        for trial in data:  # trial: (n_channels, n_times_trial)
            # 1) X：按 time 展开到 sample 维度 -> (n_times_trial, n_channels)
            X_list.append(trial.T)

            # 2) Y：对应的 time 和 class/baseline label -> (n_times_trial, 2)
            labels = make_trial_label(
                cls_label=cls_label[cls_],
                non_cls_label=cls_label[baseline_label],
                trial=trial,
                times=times,
            )
            Y_list.append(labels)

    X = np.concatenate(X_list, axis=0)  # (N_total_samples, n_channels)
    Y = np.concatenate(Y_list, axis=0)  # (N_total_samples, 2)

    return X, Y, cls_label


def make_metadata(
    cls_data: dict[str, np.ndarray],
    cls_label: dict[str, np.int_],
    times: np.ndarray,
    baseline_label: str = "*Pre-stimulus",
) -> np.ndarray:
    """Make metadata for all samples."""
    nbaseline = np.sum(times <= 0)
    ntime_pretrial = np.sum(times > 0)
    metadata = []
    n_total_trials = 0
    for cls_, data in cls_data.items():
        n_sample = data.shape[0]
        n_channel = data.shape[1]
        this = {
            "class_name": cls_,
            "class_label": cls_label[cls_],
            "n_channels": n_channel,
            "n_samples": n_sample,
            "n_times/sample": ntime_pretrial,
            "n_times": ntime_pretrial * n_sample,
        }
        metadata.append(this)
        n_total_trials += n_sample

    baseline = {
        "class_name": baseline_label,
        "class_label": cls_label[baseline_label],
        "n_channels": n_channel,
        "n_samples": n_total_trials,
        "n_times/sample": nbaseline,
        "n_times": nbaseline * n_total_trials,
    }
    metadata.append(baseline)
    return pd.DataFrame(metadata)


def prepare_input(
    dtype: str,
    pick="all",
    by="superclass_level0",
    baseline_label="*Pre-stimulus",
) -> dict:
    """Prepare the input data and plot.

    dtype: str
        "meg" or "eeg"
    pick: str
        Sensor of interest.
    by: str
        Metadata column to group by.
    """
    evo = load_evo(dtype)
    times = get_times(evo)
    cls_data = get_cls_data(
        evo,
        by=by,
        soi=pick,
    )
    balanced_cls_data = balance_cls_data(cls_data)
    X, Y, cls_label = make_input(
        balanced_cls_data, times, baseline_label=baseline_label
    )
    metadata = make_metadata(
        balanced_cls_data, cls_label, times, baseline_label=baseline_label
    )
    preproc_data = {
        "X": X,
        "Y": Y,
        "metadata": metadata,
        "times": times,
        "cls_label": cls_label,
    }
    return preproc_data


if __name__ == "__main__":
    data = {}

    for dtype in ["meg", "eeg"]:
        res = prepare_input(
            dtype=dtype,
            pick="all",
            by="superclass_level0",
            baseline_label=r"*baseline",
        )
        data[dtype] = res

    plot_labels = {}
    for k, v in data["eeg"]["cls_label"].items():
        new_k = k.split(maxsplit=1)[0]
        new_k = new_k.replace(",", "").replace(" ", "")
        plot_labels[new_k] = v
    data["plot_labels"] = plot_labels
    dump(data, SAVE_DIR / "cebra_input_meeg.pkl")
