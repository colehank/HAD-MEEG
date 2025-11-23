import os
from mne_bids import BIDSPath
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ICAPath:
    base: str
    dtypr: str
    tsv: Path
    all_png: Path
    comp_pngs: dict[str, Path]


@dataclass
class ICALabels:
    auto_labels: list[str]
    manualed: bool = False
    _manual_labels: Optional[list[str]] = field(default=None, repr=False)

    @property
    def manual_labels(self):
        return self._manual_labels

    @manual_labels.setter
    def manual_labels(self, value):
        if value is not None:
            if len(value) != len(self.auto_labels):
                raise ValueError("manual_labels dismatch auto_labels's length")
            self.manualed = True
        else:
            self.manualed = False

        self._manual_labels = value


class ICAData:
    def __init__(self, bids: BIDSPath):
        self.ica_path: ICAPath = self._get_ica_path(bids)

        self.candidate_labels = [
            "brain",
            "muscle artifact",
            "eye blink",
            "heart beat",
            "line noise",
            "channel noise",
            "other",
            "brain/other",
            "eye movement",
        ]

    @property
    def ica_labels(self) -> ICALabels:
        auto_labels = self._get_auto_labels()
        labels = ICALabels(
            auto_labels=auto_labels,
        )
        labels.manual_labels = getattr(self, "manual_labels", None)
        return labels

    def put_manual_labels(self, manual_labels: list[str]):
        if set(manual_labels).issubset(self.candidate_labels):
            self.manual_labels = manual_labels
            self.ica_labels
        else:
            raise ValueError(f"labels must belongs to {self.candidate_labels}")

    def _get_ica_path(self, bids: BIDSPath) -> ICAPath:
        sub = bids.subject
        ses = bids.session
        dtype = bids.datatype
        basename = bids.basename
        preproc_dir = bids.root / "derivatives" / "preproc"
        this_dir = preproc_dir / f"sub-{sub}" / f"ses-{ses}" / dtype
        base_name = this_dir / f"{basename}_desc-ica_{dtype}"
        ica_tsv = base_name.with_suffix(".tsv")
        ica_all = base_name.with_suffix(".png")
        ica_comps = os.listdir(base_name)
        ica_comps = {
            i.split(".")[0].split("-")[1]: base_name / i
            for i in ica_comps
            if i.endswith(".png") and "comp" in i
        }

        return ICAPath(
            base=str(base_name),
            dtypr=dtype,
            tsv=ica_tsv,
            all_png=ica_all,
            comp_pngs=ica_comps,
        )

    def _get_auto_labels(self) -> ICALabels:
        import pandas as pd

        label = pd.read_csv(self.ica_path.tsv, sep="\t")
        label = label["ic_type"].tolist()
        return label
