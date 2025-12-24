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
    """Configuration for M/EEG BIDS dataset."""

    bids_root: DirectoryPath
    derivatives_root: DirectoryPath | None = None
    results_root: DirectoryPath | None = None

    subjects: list[str] = Field(default_factory=list)
    sessions: list[str | None] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)

    datatypes: list[str] = Field(default_factory=lambda: ["meg", "eeg"])
    meg_raw_extension: str | None = None
    eeg_raw_extension: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="MEEG_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- validators ----------

    @field_validator("derivatives_root", mode="before")
    @classmethod
    def set_default_derivatives_root(cls, v, info):
        """If derivatives_root is not set, default to bids_root/derivatives."""
        if v is not None:
            return v
        bids_root = info.data.get("bids_root")
        if not bids_root:
            return None
        return Path(bids_root) / "derivatives"

    @field_validator("results_root", mode="before")
    @classmethod
    def set_default_results_root(cls, v, info):
        """If results_root is not set, default to bids_root../HAD-MEEG_results."""
        if v is not None:
            return v
        bids_root = info.data.get("bids_root")
        if not bids_root:
            return None
        return Path(bids_root).parent / "HAD-MEEG_results"

    @model_validator(mode="after")
    def auto_fill_entities(self):
        """Auto-fill subjects, sessions, tasks, and runs if they are not set."""
        if not self.subjects:
            self.subjects = get_entity_vals(self.bids_root, "subject")

        if not self.sessions:
            sessions = get_entity_vals(self.bids_root, "session")
            self.sessions = sessions or [None]

        if not self.tasks:
            self.tasks = get_entity_vals(self.bids_root, "task")

        if not self.runs:
            self.runs = get_entity_vals(self.bids_root, "run")

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
                preped_runs = self.preprocessed[sub][ses]
                for bp, prep in zip(bp_list, preped_runs):
                    rows.append(
                        {
                            "subject": sub,
                            "session": ses,
                            "task": bp.task,
                            "run": bp.run,
                            "datatype": bp.datatype,
                            "raw": str(bp.fpath),
                            "preprocessed": str(prep),
                        },
                    )
        df = pd.DataFrame(rows)
        df["epochs"] = df.apply(
            lambda row: str(
                self.derivatives_root
                / "epochs"
                / f"sub-{row['subject']}_epo_{row['datatype']}.fif"
            ),
            axis=1,
        )
        return df

    @cached_property
    def preprocessed(self) -> dict:
        to_return = {}
        preproc_dir = self.derivatives_root / "preproc"
        source = self.source
        for sub in source:
            to_return[sub] = {}
            for ses in source[sub]:
                to_return[sub][ses] = []
                for bids in source[sub][ses]:
                    fp = (
                        preproc_dir
                        / f"sub-{bids.subject}"
                        / f"ses-{bids.session}"
                        / bids.session
                        / f"{bids.basename}_preproc_{bids.session}.fif"
                    )
                    if fp.exists():
                        to_return[sub][ses].append(fp)
        return to_return

    @cached_property
    def epoched(self) -> dict:
        to_return = {}
        epo_dir = self.derivatives_root / "epoch"
        for sub in self.subjects:
            to_return[sub] = {}
            if "meg" in self.source[sub].keys():
                fp_meg = epo_dir / f"sub-{sub}_epo_meg.fif"
                if fp_meg.exists():
                    to_return[sub]["meg"] = fp_meg
            if "eeg" in self.source[sub].keys():
                fp_eeg = epo_dir / f"sub-{sub}_epo_eeg.fif"
                if fp_eeg.exists():
                    to_return[sub]["eeg"] = fp_eeg
        return to_return

    @cached_property
    def source_bids_list(self) -> list[BIDSPath]:
        """Create a list of all source BIDSPaths."""
        bids_list: list[BIDSPath] = []
        for ses_dic in self.source.values():
            for bp_list in ses_dic.values():
                bids_list.extend(bp_list)
        return bids_list

    @cached_property
    def detailed_events(self) -> dict[str, Path]:
        to_return = {}
        events_dir = self.derivatives_root / "detailed_events"
        for sub in self.subjects:
            fp = events_dir / f"sub-{sub}_events.tsv"
            if fp.exists():
                to_return[sub] = fp
        return to_return

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
        if datatype == "meg":
            return self.meg_raw_extension
        if datatype == "eeg":
            return self.eeg_raw_extension
        return None
