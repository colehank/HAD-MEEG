# %%
from __future__ import annotations
import pandas as pd
from tqdm.auto import tqdm
import sys

# original events cols: ['onset', 'duration', 'trial_type', 'value', 'sample']
sys.path.append("..")
from src.config import DataConfig


# %%
def get_events(bids):
    ev_fp = get_event_path(bids)
    df = pd.read_csv(ev_fp, sep="\t")
    df = df.drop_duplicates()
    return df


def get_event_path(bids):
    dirname = bids.directory
    basename = bids.basename
    ev_fp = dirname / f"{basename}_events.tsv"
    if not ev_fp.exists():
        raise FileNotFoundError(f"Events file not found: {ev_fp}")
    return ev_fp


def get_detailed_events(bids):
    sub = bids.subject
    ses = bids.session
    task = bids.task
    run = bids.run
    dirname = cfg.derivatives_root / "detailed_events"
    ev_fp = dirname / f"sub-{sub}_events.tsv"
    if not ev_fp.exists():
        raise FileNotFoundError(f"Detailed events file not found: {ev_fp}")
    df = pd.read_csv(ev_fp, sep="\t")
    df.rename(columns={"event_name": "trial_type"}, inplace=True)
    return df[(df["task"] == task) & (df["run"] == int(run)) & (df["session"] == ses)]


def update_events(ev: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    """Update the events DataFrame `ev` with information from `dev` DataFrame.

    The function matches rows based on 'video on' trial types and updates
    the columns in `ev` with corresponding values from `dev`, excluding
    the 'trial_type' column.

    Args:
        ev (pd.DataFrame): Original events DataFrame to be updated.
        dev (pd.DataFrame): Detailed events DataFrame with new information.

    Returns:
        pd.DataFrame: Updated events DataFrame.
    """
    ev_ = ev.copy()
    dev_ = dev.copy()
    idx_a = ev_[ev_["trial_type"] == "video on"].index
    idx_b = dev_[dev_["trial_type"] == "video_on"].index
    assert len(idx_a) == len(idx_b), "两边 video_on 的行数不一致！"
    cols_b = dev_.columns.difference(["trial_type"])
    for col in cols_b:
        ev_.loc[idx_a, col] = dev_.loc[idx_b, col].values
    ev_ = ev_[
        [
            "onset",
            "duration",
            "trial_type",
            "value",
            "sample",
            "video_id",
            "class_id",
            "super_class_id",
            "class_name",
            "super_class_name",
            "is_resp",
            "stim_is_sports",
            "resp_is_sports",
            "resp_is_right",
            "resp_time",
        ]
    ]
    return ev_


# %%
if __name__ == "__main__":
    bids_root = "../../HAD-MEEG_upload"
    cfg = DataConfig(bids_root=bids_root)
    all_bids = cfg.source_bids_list

    for bids in tqdm(all_bids):
        ev_path = get_event_path(bids)
        ev = get_events(bids)
        dev = get_detailed_events(bids)
        new_ev = update_events(ev, dev)
        new_ev.to_csv(ev_path, sep="\t", index=False, na_rep="n/a", encoding="utf-8")
# %%
