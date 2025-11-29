# %%
from __future__ import annotations

import mne
import pandas as pd
import numpy as np
from typing import Iterable, Tuple, List
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


# %%
class EvokedSet:
    """Evoked container with mne.Epochs.metadata like management.

    Parameters
    ----------
    evokeds : list[mne.Evoked]
        A list of mne.Evoked objects.
    metadata : pd.DataFrame
        The number of rows must match the number of evokeds, with each row corresponding to one Evoked.
    """

    def __init__(self, evokeds: List, metadata: pd.DataFrame) -> None:
        # --- Basic checks ---
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        if len(evokeds) != len(metadata):
            raise ValueError(
                f"evokeds length is {len(evokeds)}, "
                f"metadata length is {len(metadata)}, they must match."
            )

        self._evokeds: List = list(evokeds)
        # Reset index to ensure clean row numbers after boolean indexing/iloc
        self._metadata: pd.DataFrame = metadata.reset_index(drop=True)

    # --------- Properties ---------

    @property
    def metadata(self) -> pd.DataFrame:
        """Metadata corresponding one-to-one with Evokeds (one row per Evoked)."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._evokeds)

    def __iter__(self):
        return iter(self._evokeds)

    def __repr__(self) -> str:
        return (
            f"<EvokedSet | n_evokeds={len(self)}, "
            f"metadata_cols={list(self.metadata.columns)}>"
        )

    # ---------  ---------

    def _resolve_indices(self, item) -> np.ndarray:
        """Convert various index types to an array of integer indices."""
        n = len(self)
        if isinstance(item, (pd.Series, pd.Index)):
            arr = np.asarray(item.values)
        else:
            arr = np.asarray(item)

        # Boolean mask
        if arr.dtype == bool:
            if arr.shape[0] != n:
                raise IndexError(
                    f"Boolean index length is {arr.shape[0]}, "
                    f"but EvokedSet length is {n}."
                )
            return np.flatnonzero(arr)

        # Other cases are treated as integer indices (scalar or array)
        return arr

    def __getitem__(self, item) -> "EvokedSet":
        """Support:
        - Integer: evo[3]
        - Slice: evo[1:10:2]
        - Boolean mask: evo[evo.metadata["col"] == 2]
        - List/array of integers: evo[[0, 2, 5]]
        """
        # slice / int cases
        if isinstance(item, (slice, int)):
            new_evokeds = self._evokeds[item]
            new_meta = self._metadata.iloc[item].reset_index(drop=True)

            # int index returns a single object
            if isinstance(item, int):
                fil_evo = EvokedSet([new_evokeds], new_meta.to_frame().T)
                return fil_evo.evoked

            fil_evo = EvokedSet(new_evokeds, new_meta)
            return fil_evo.evoked

        # Other cases: boolean / integer array/list/Series
        indices = self._resolve_indices(item)
        new_evokeds = [self._evokeds[i] for i in indices]
        new_meta = self._metadata.iloc[indices].reset_index(drop=True)
        fil_evo = EvokedSet(new_evokeds, new_meta)
        return fil_evo.evoked

    @property
    def evoked(self) -> mne.Evoked | List[mne.Evoked]:
        """If only one Evoked inside, return it directly; else return the list."""
        if len(self._evokeds) == 1:
            return self._evokeds[0]
        return self._evokeds


class EpoToEvo(EvokedSet):
    """Epochs container with mne.Epochs.metadata like management.

    Parameters
    ----------
    epochs : list[mne.Epochs]
        A list of mne.Epochs objects.
    """

    def __init__(
        self,
        epochs: list[mne.Epochs],
        by_col: str,
        remaining_cols: list[str] | None = None,
        n_jobs: int = 1,
    ) -> None:
        if isinstance(remaining_cols, str):
            remaining_cols = [remaining_cols]
        evo, metadata = self._epo2evo_by_meta(epochs, by_col, remaining_cols, n_jobs)
        super().__init__(evo, metadata)

    def _average_one_value(
        self,
        val,
        epo: mne.Epochs,
        by_col: str,
        remaining_cols: list[str] | None = None,
    ) -> Tuple[mne.Evoked, dict]:
        """给单个 metadata 取值算一个 Evoked，并返回一行 meta 信息。"""
        mask = epo.metadata[by_col] == val
        sel_epochs = epo[mask]

        evoked = sel_epochs.average()
        evoked.comment = f"{by_col}-{val}"

        meta_row: dict = {
            "n_avg": len(sel_epochs),
            by_col: val,
        }

        if remaining_cols:
            first_row = sel_epochs.metadata.iloc[0]
            for col in remaining_cols:
                meta_row[col] = first_row[col]

        return evoked, meta_row

    def _epo2evo_by_meta(
        self,
        epo: mne.Epochs,
        by_col: str,
        remaining_cols: list[str] | None = None,
        n_jobs: int = 1,
    ):
        """According to a metadata column, average epochs into evokeds.

        Parameters
        ----------
        epo : mne.Epochs
            The epochs to be averaged.
        by_col : str
            The metadata column name to group by.
        remaining_cols : list[str] | None
            Other metadata columns to keep in the output meta DataFrame.
        n_jobs : int
            Number of parallel jobs.

        Returns
        -------
        evokeds : list[mne.Evoked]
            Each group corresponds to one Evoked.
        meta_df : pd.DataFrame
            Each Evoked corresponds to one row of meta information
            (including n_avg, by_col, and remaining_cols).
        """
        unique_vals = epo.metadata[by_col].unique()
        iterator: Iterable = unique_vals
        with tqdm_joblib(
            total=len(unique_vals), desc=f"Computing evokeds by {by_col}", leave=False
        ):
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._average_one_value)(val, epo, by_col, remaining_cols)
                for val in iterator
            )
        evokeds, meta_rows = zip(*results)
        meta_df = pd.DataFrame(list(meta_rows))

        return list(evokeds), meta_df
