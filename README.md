# HAD-MEEG: Human Action Dataset of Magnetoencephalography and Electroencephalography

This repository provides the codebase for preprocessing, validation, and visualization associated with the *[A large-scale MEG and EEG dataset for human action recognition](https://reqbin.com/)*.
![Pipeline Diagram](resources/overview.png)

## Installation
Code is tested with **Python 3.11+** on Linux and MacOS.

```bash
git clone https://github.com/colehank/HAD-MEEG.git
cd HAD-MEEG
```

This project uses [`uv`]((https://docs.astral.sh/uv/getting-started/installation/)) for Python dependency management.
```bash
uv sync
```

Or, you can still install the dependencies with pip.

```bash
# using venv
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# or using conda
# conda create -n had-meeg python=3.11
# conda activate had-meeg

pip install -r requirements.txt
```

## Configuration

1. Download `HAD-MEEG` from OpenNeuro [dsxxxxxx]()

1. Rename `.env_example` to  `.env`.

2. Edit `.env` and set `MEEG_BIDS_ROOT` to the path of the `HAD-MEEG`'s root directory on your machine.  The `.env` file is loaded at runtime to locate the dataset.

After this, you should be able to run all analyses in `scripts/`.

## Usage

### Running Analysis Scripts
All preprocessing and analysis scripts are in the `scripts/` directory. Run them from the project root:

```bash
# Using uv
uv run scripts/step-*.py

# Or using Python directly
python scripts/step-*.py
```

### Working with Epochs

#### Filtering by Action Class
Trial metadata is embedded in `mne.Epochs.metadata`, allowing you to filter epochs by action class or hierarchical categories.

**Filter by specific action:**
```python
import mne

epo = mne.read_epochs("HAD-MEEG/derivatives/epochs/sub-01_epo_meg.fif")
surfing_epo = epo[epo.metadata["class_name"] == "Surfing"]
```

**Filter by hierarchical categories:**
```python
import mne

epo = mne.read_epochs("HAD-MEEG/derivatives/epochs/sub-01_epo_meg.fif")

# Level 1: Participating in Sports, Exercise, or Recreation
sports_epo = epo[epo.metadata["superclass_level1"] == "Participating in Sports, Exercise, or Recreation"]

# Level 2: Participating in water sports
water_sports_epo = sports_epo[sports_epo.metadata["superclass_level2"] == "Participating in water sports"]

# Specific class: Surfing
surfing_epo = water_sports_epo[water_sports_epo.metadata["class_name"] == "Surfing"]
```
