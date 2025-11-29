from .epoching import Epocher
import mne
from mne.io.base import BaseRaw
from joblib import Parallel, delayed
from typing import Sequence
from tqdm_joblib import tqdm_joblib
# def align_meg_head(
#     raws: Sequence[BaseRaw | mne.Epochs],
#     ref_idx:int = 0,
# ) -> list[BaseRaw | mne.Epochs]:
#     """Align CTF MEG data to head position using specified method."""
#     ref_info = raws[ref_idx].info
#     for raw in tqdm(
#         raws[1:], desc="Aligning MEG heads' positions", position=1, leave=False
#     ):
#         this_info = raw.info
#         map = mne.forward._map_meg_or_eeg_channels(
#             ref_info, this_info, mode="accurate", origin=(0.0, 0.0, 0.04)
#         )
#         ori_data = raw.get_data(picks="meg")
#         aligned_data = map @ ori_data

#         raw._data = aligned_data
#         raw.info.update(
#             {
#                 "dev_head_t": ref_info["dev_head_t"],
#             }
#         )
#     return raws


def _align_single(
    raw: BaseRaw | mne.Epochs,
    ref_info: mne.Info,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.04),
) -> BaseRaw | mne.Epochs:
    """对单个 Raw/Epochs 做头位对齐，返回新的对象。"""

    # 每个 worker 用自己的拷贝，避免并发修改同一个对象
    raw = raw.copy()
    this_info = raw.info

    # ref_info 也 copy 一份，避免 _map_meg_or_eeg_channels / make_ad_hoc_cov
    # 在内部对 info 做任何 inplace 改动时产生竞态
    ref_info = ref_info.copy()

    mapping = mne.forward._map_meg_or_eeg_channels(
        ref_info,
        this_info,
        mode="accurate",
        origin=origin,
    )

    ori_data = raw.get_data(picks="meg")
    aligned_data = mapping @ ori_data

    # 与你原来的用法保持一致
    raw._data = aligned_data
    raw.info["dev_head_t"] = ref_info["dev_head_t"]

    return raw


def align_meg_head(
    raws: Sequence[BaseRaw | mne.Epochs],
    ref_idx: int = 0,
    n_jobs: int = 1,
    prefer: str = "threads",  # 用线程！避免 Info 被 pickle 之后出事
) -> list[BaseRaw | mne.Epochs]:
    """并行对齐 CTF MEG 数据到统一 head position。"""

    raws = list(raws)
    ref_raw = raws[ref_idx]
    ref_info = ref_raw.info

    # 除了 ref 以外的索引
    other_indices = [i for i in range(len(raws)) if i != ref_idx]
    other_raws = [raws[i] for i in other_indices]

    with tqdm_joblib(total=len(other_raws), desc="Aligning MEG head positions"):
        aligned_others = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            prefer=prefer,
        )(delayed(_align_single)(raw, ref_info) for raw in other_raws)

    # 按原来的顺序放回
    out = list(raws)
    for idx, aligned in zip(other_indices, aligned_others):
        out[idx] = aligned
    out[ref_idx] = ref_raw  # 参考那一条保持不变

    return out


def concat_epochs(
    epochs_list: list[mne.Epochs],
    align_head: bool = True,
    n_jobs: int = 1,
) -> mne.Epochs:
    """Concatenate a list of mne.Epochs into one."""
    if align_head:
        epochs_list = align_meg_head(epochs_list, n_jobs=n_jobs)
    concat_epochs = mne.concatenate_epochs(
        epochs_list,
        add_offset=True,
    )
    return concat_epochs


__all__ = ["Epocher", "align_meg_head", "concat_epochs"]
