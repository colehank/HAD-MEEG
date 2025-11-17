from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
from mne_bids import BIDSPath
from mne_bids import get_entity_vals
from pydantic import DirectoryPath
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class DataConfig(BaseSettings):
    """Configuration for M/EEG BIDS dataset.

    Parameters
    ----------
    bids_root : DirectoryPath
        Path to the BIDS root directory.
    derivatives_root : Optional[DirectoryPath], optional
        Path to the derivatives directory. If None, defaults to `bids_root/derivatives`.
    subjects : list[str], optional
        List of subject IDs to include. If empty, will be auto-filled from BIDS structure.
    sessions : list[Optional[str]], optional
        List of session IDs to include. If empty, will be auto-filled from BIDS structure.
    tasks : list[str], optional
        List of task names to include. If empty, will be auto-filled from BIDS structure.
    runs : list[str], optional
        List of run numbers to include. If empty, will be auto-filled from BIDS structure.
    datatypes : list[str], optional
        List of data types to include (e.g., "meg", "eeg"). Defaults to ["meg", "eeg"].
    meg_raw_extension : Optional[str], optional
        File extension for MEG raw data files (e.g., ".fif"). Will be loaded from environment if not set.
    eeg_raw_extension : Optional[str], optional
        File extension for EEG raw data files (e.g., ".set"). Will be loaded from environment if not set.

    Returns
    -------
    MEEGConfig
        An instance of the MEEGConfig class with the specified settings.
    """

    bids_root: DirectoryPath
    derivatives_root: DirectoryPath | None = None

    subjects: list[str] = Field(default_factory=list)
    sessions: list[str | None] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)

    datatypes: list[str] = Field(default_factory=lambda: ['meg', 'eeg'])
    meg_raw_extension: str | None = None
    eeg_raw_extension: str | None = None

    model_config = SettingsConfigDict(
        env_prefix='MEEG_',
        env_file='.env',
        case_sensitive=False,
        extra='ignore',
    )

    # ---------- validators ----------

    @field_validator('derivatives_root', mode='before')
    @classmethod
    def set_default_derivatives_root(cls, v, info):
        """If derivatives_root is not set, default to bids_root/derivatives."""
        if v is not None:
            return v
        bids_root = info.data.get('bids_root')
        if not bids_root:
            return None
        return Path(bids_root) / 'derivatives'

    @model_validator(mode='after')
    def auto_fill_entities(self):
        """Auto-fill subjects, sessions, tasks, and runs if they are not set."""
        if not self.subjects:
            self.subjects = get_entity_vals(self.bids_root, 'subject')

        if not self.sessions:
            sessions = get_entity_vals(self.bids_root, 'session')
            self.sessions = sessions or [None]

        if not self.tasks:
            self.tasks = get_entity_vals(self.bids_root, 'task')

        if not self.runs:
            self.runs = get_entity_vals(self.bids_root, 'run')

        return self

    @cached_property
    def source(self) -> dict[str, dict[str | None, list[BIDSPath]]]:
        """Find and cache the source BIDSPaths based on the configuration."""
        return self._find_source()

    @cached_property
    def source_df(self) -> pd.DataFrame:
        """Create a DataFrame summarizing the source BIDSPaths."""
        rows: list[dict[str, Any]] = []
        for sub, ses_dic in self.source.items():
            for ses, bp_list in ses_dic.items():
                for bp in bp_list:
                    rows.append(
                        {
                            'subject': sub,
                            'session': ses,
                            'task': bp.task,
                            'run': bp.run,
                            'datatype': bp.datatype,
                            'raw_fpath': str(bp.fpath),
                        },
                    )
        return pd.DataFrame(rows)

    @cached_property
    def source_bids_list(self) -> list[BIDSPath]:
        """Create a list of all source BIDSPaths."""
        bids_list: list[BIDSPath] = []
        for ses_dic in self.source.values():
            for bp_list in ses_dic.values():
                bids_list.extend(bp_list)
        return bids_list

    def _find_source(self) -> dict[str, dict[str | None, list[BIDSPath]]]:
        """Find all BIDSPaths matching the configured subjects, sessions, tasks, runs, and datatypes."""
        bids_root = self.bids_root
        result: dict[str, dict[str | None, list[BIDSPath]]] = {}

        for sub in self.subjects:
            result[sub] = {}
            for ses in self.sessions:
                result[sub][ses] = []
                for task in self.tasks:
                    for run in self.runs:
                        for datatype in self.datatypes:
                            ext = self._get_extension_for_datatype(datatype)
                            bp = BIDSPath(
                                root=bids_root,
                                subject=sub,
                                session=ses,
                                task=task,
                                run=run,
                                datatype=datatype,
                                extension=ext,
                            )
                            if bp.fpath.exists():
                                result[sub][ses].append(bp)
        return result

    def _get_extension_for_datatype(self, datatype: str) -> str | None:
        """Get the file extension for the given datatype."""
        if datatype == 'meg':
            return self.meg_raw_extension
        if datatype == 'eeg':
            return self.eeg_raw_extension
        return None
