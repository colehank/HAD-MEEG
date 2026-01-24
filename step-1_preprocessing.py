from __future__ import annotations

from pathlib import Path
import json
from src import DataConfig, PrepPipe, AnalyseConfig
from src.prep import run_ica_app

cfg_data = DataConfig()
cfg_anal = AnalyseConfig()

ALL_BIDS = cfg_data.source_bids_list
N_JOBS: int = cfg_anal.n_jobs
RANDOM_STATE: int = cfg_anal.random_state
USE_CUDA: bool = cfg_anal.use_cuda
TEMP_ANNOTS_FILE = Path(
    "./ica_labels.json"
)  # Automatic generated ICA label in following steps


def load_manual_ica_labels(
    annot_path: Path,
    bids_list,
) -> tuple[bool, dict | None]:
    """
    Try to load manually checked ICA labels.

    Returns:
        (manual_checked, manual_labels)
        manual_checked: True if labels are available and match BIDS list
        manual_labels:  dict of labels if available, otherwise None
    """
    if not annot_path.exists():
        return False, None

    with annot_path.open("r", encoding="utf-8") as f:
        manual_labels = json.load(f)

    bids_names = {bids.basename for bids in bids_list}
    label_names = set(manual_labels.keys())

    if bids_names == label_names:
        return True, manual_labels

    # File exists but does not match current BIDS list, treat as not checked
    return False, None


if __name__ == "__main__":
    # Initialize data config and preprocessing pipeline
    batch_pipe = PrepPipe(
        bids_list=ALL_BIDS,
        use_cuda=USE_CUDA,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )

    # Try to load existing ICA labels (if any)
    manual_checked, manual_ica_labels = load_manual_ica_labels(
        TEMP_ANNOTS_FILE,
        ALL_BIDS,
    )

    # Step 1: Preprocessing before manual ICA labeling (skip ICA regression)
    if not manual_checked:
        batch_pipe.run(
            manual_ica_checked=False,
            skip_ica=True,  # Will skip if *_ica_meg/eeg.fif already exists
        )

    # Step 2: Manual ICA label inspection
    if not manual_checked:
        run_ica_app(
            process_file=TEMP_ANNOTS_FILE,
            bids_list=ALL_BIDS,
        )
        # Reload labels after manual inspection
        manual_checked, manual_ica_labels = load_manual_ica_labels(
            TEMP_ANNOTS_FILE,
            ALL_BIDS,
        )
        if not manual_checked:
            raise RuntimeError(
                "ICA labels file does not match BIDS list after manual "
                "checking. Please verify ica_labels.json."
            )

    # Step 3: Run preprocessing again using manual ICA labels and regress artifacts
    batch_pipe.run(
        skip_ica=False,
        skip_raw=True,  # Will skip if *_preproc_meg/eeg.fif already exists
        manual_ica_checked=True,
        manual_labels=manual_ica_labels,
        regress=True,  # Regress artifact ICs from raw data
    )
