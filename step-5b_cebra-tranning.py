from joblib import load
from src import DataConfig
from src.cebra import ModelZoo, _norm

cfg = DataConfig()
SAVE_DIR = cfg.results_root / "cebra"
MODEL_DIR = SAVE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0"
RANDOM_SEED = 42

data = load(SAVE_DIR / "cebra_input_meeg.pkl")
meg = data["meg"]
eeg = data["eeg"]
labels = data["plot_labels"]


if __name__ == "__main__":
    # define models and plot train data
    model_types = ["class_only", "class_time"]
    max_it = 20_000
    wb = False  # rm baseline data

    meg_train = ModelZoo(
        X=_norm(meg["X"]),
        Y=meg["Y"],
        model_types=model_types,
        rm_label=labels.get("*baseline", None),
        save_dir=SAVE_DIR / "models" / "meg" / f"it-{max_it}_wb-{wb}",
        random_seed=RANDOM_SEED,
        max_iterations=20_000,
        output_dimension=3,
    )
    eeg_train = ModelZoo(
        X=_norm(eeg["X"]),
        Y=eeg["Y"],
        model_types=model_types,
        rm_label=labels.get("*baseline", None),
        save_dir=SAVE_DIR / "models" / "eeg" / f"it-{max_it}_wb-{wb}",
        random_seed=RANDOM_SEED,
        max_iterations=20_000,
        output_dimension=3,
    )
    meg_train.fit_all(with_shuffle=True)
    eeg_train.fit_all(with_shuffle=True)
