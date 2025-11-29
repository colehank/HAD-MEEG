# %%
from pathlib import Path
from joblib import load
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder
import mne
from dataclasses import dataclass
import logging

import cebra
from cebra import CEBRA, plot_embedding
import cebra.integrations.plotly
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap


# %%
def prepare_neural_data(
    balanced_evos: dict[str, mne.Evoked],
    time_samples: int,
    time_start: float,
    time_end: float,
) -> dict:
    """Prepare neural data for CEBRA training."""
    neural_data = []
    for evo in balanced_evos.values():
        neural_data.append(evo.T)

    neural_data = np.array(neural_data)
    neural_data = neural_data.reshape(-1, neural_data.shape[-1])

    # Normalize data
    neural_data = (neural_data - neural_data.min(axis=0)) / (
        neural_data.max(axis=0) - neural_data.min(axis=0)
    )
    neural_data = torch.tensor(neural_data)

    # Create labels
    labels = list(balanced_evos.keys())
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    y = np.repeat(numeric_labels, time_samples)

    # Create time index
    time_index = np.linspace(time_start, time_end, time_samples)
    label_with_time = np.tile(time_index, len(labels))
    time_label = np.array(np.tile(time_index, len(labels)))
    combined_labels = np.vstack((label_with_time, y)).T

    return {
        "neural_data": neural_data,
        "class_names": labels,
        "numeric_labels": numeric_labels,
        "time_points": time_index,
        "class_labels": y,
        "time_labels": time_label,
        "combined_labels": combined_labels,
        "label_encoder": label_encoder,
    }


@dataclass
class Config:
    # Output directories
    DATA_ROOT: str = "../HAD-MEEG_results/cebra"
    MEG_ROOT: str = f"{DATA_ROOT}/eeg"

    DATA_DIR: str = f"{MEG_ROOT}/data"
    MODELS_DIR: str = f"{MEG_ROOT}/models"
    FIGURES_DIR: str = f"{MEG_ROOT}/figures"
    MEG__ROOT: str = f"{DATA_ROOT}/eeg_data"
    ANIMATIONS_DIR: str = f"{MEG_ROOT}/animations"
    # CEBRA model parameters
    MAX_ITERATIONS: int = 50000
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 3e-4
    OUTPUT_DIMENSION: int = 3
    TIME_OFFSETS: int = 10

    # Data processing parameters
    MAX_EVOS_PER_CLASS: int = 20
    TIME_SAMPLES: int = 226
    TIME_START: float = -100
    TIME_END: float = 800

    # Visualization parameters
    FIGURE_DPI: int = 300
    BASELINE: int = 100
    STIM_ONSET: int = 800
    CMAP1: str = "Greens_r"
    CMAP2: str = "magma_r"
    ANIMATION_FPS: int = 30
    ANIMATION_FRAMES: int = 360


def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    # Create logs directory
    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "cebra_analysis.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# %%
class DataLoader:
    """Handles loading and preprocessing of neural data."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_evoked_data(self) -> list:
        """Load evoked responses from file."""
        try:
            self.logger.info(f"Loading evoked data from {self.config.GRAND_EVOS_PATH}")
            return joblib.load(self.config.GRAND_EVOS_PATH)
        except FileNotFoundError:
            self.logger.error(
                f"Evoked data file not found: {self.config.GRAND_EVOS_PATH}"
            )
            raise

    def load_class_metadata(self) -> pd.DataFrame:
        """Load and filter class metadata."""
        try:
            self.logger.info(f"Loading class metadata from {self.config.CLASSES_PATH}")
            classes = pd.read_csv(self.config.CLASSES_PATH)
            return classes[classes["task"] == self.config.TASK_FILTER]
        except FileNotFoundError:
            self.logger.error(
                f"Class metadata file not found: {self.config.CLASSES_PATH}"
            )
            raise

    def create_class_mappings(self, classes: pd.DataFrame) -> tuple[dict, dict]:
        """Create mappings between super classes and class IDs/names."""
        super_classes = classes["super_class"].unique()

        class_id_map = {
            super_cls: classes[classes["super_class"] == super_cls]["class_id"]
            .unique()
            .tolist()
            for super_cls in super_classes
        }

        class_name_map = {
            super_cls: classes[classes["super_class"] == super_cls]["class"]
            .unique()
            .tolist()
            for super_cls in super_classes
        }

        return class_id_map, class_name_map


class DataPreprocessor:
    """Handles data preprocessing and preparation for CEBRA."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def group_evokeds_by_superclass(self, all_evos: list, class_id_map: dict) -> dict:
        """Group evoked responses by super class."""
        super_evos = {}
        for super_cls, class_ids in class_id_map.items():
            super_evos[super_cls] = []
            for evo in all_evos:
                if evo.comment in class_ids:
                    super_evos[super_cls].append(evo)
        return super_evos

    def balance_evokeds(self, super_evos: dict) -> dict:
        """Balance evoked responses across classes."""
        balanced_evos = {}
        for super_cls, evos in super_evos.items():
            if len(evos) > self.config.MAX_EVOS_PER_CLASS:
                balanced_evos[super_cls] = mne.grand_average(
                    evos[: self.config.MAX_EVOS_PER_CLASS]
                )
                self.logger.info(
                    f"Balanced {super_cls}: {len(evos)} -> {self.config.MAX_EVOS_PER_CLASS}"
                )
        return balanced_evos

    def prepare_neural_data(self, balanced_evos: dict) -> dict:
        """Prepare neural data for CEBRA training."""
        neural_data = []
        for evo in balanced_evos.values():
            neural_data.append(evo.data.T)

        neural_data = np.array(neural_data)
        neural_data = neural_data.reshape(-1, neural_data.shape[-1])

        # Normalize data
        neural_data = (neural_data - neural_data.min(axis=0)) / (
            neural_data.max(axis=0) - neural_data.min(axis=0)
        )
        neural_data = torch.tensor(neural_data)

        # Create labels
        labels = list(balanced_evos.keys())
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
        y = np.repeat(numeric_labels, self.config.TIME_SAMPLES)

        # Create time index
        time_index = np.linspace(
            self.config.TIME_START, self.config.TIME_END, self.config.TIME_SAMPLES
        )
        label_with_time = np.tile(time_index, len(labels))
        time_label = np.array(np.tile(time_index, len(labels)))
        combined_labels = np.vstack((label_with_time, y)).T

        return {
            "neural_data": neural_data,
            "class_names": labels,
            "numeric_labels": numeric_labels,
            "time_points": time_index,
            "class_labels": y,
            "time_labels": time_label,
            "combined_labels": combined_labels,
            "label_encoder": label_encoder,
        }

    def save_processed_data(
        self, data: dict, filename: str = "superclass_data.pkl"
    ) -> None:
        """Save processed data to disk."""
        save_path = Path(self.config.DATA_DIR) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, save_path)
        self.logger.info(f"Saved processed data to {save_path}")


class CEBRAModelManager:
    """Manages CEBRA model creation, training, and persistence."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def create_cebra_model(self, conditional: str = "time_delta") -> CEBRA:
        """Create a CEBRA model with specified parameters."""
        return CEBRA(
            model_architecture="offset10-model",
            batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            temperature_mode="auto",
            output_dimension=self.config.OUTPUT_DIMENSION,
            max_iterations=self.config.MAX_ITERATIONS,
            distance="cosine",
            conditional=conditional,
            device="cuda_if_available",
            verbose=True,
            time_offsets=self.config.TIME_OFFSETS,
        )

    def train_models(self, neural_data: dict) -> dict[str, CEBRA]:
        """Train multiple CEBRA models with different conditioning."""
        models = {}

        # Create models directory
        models_dir = Path(self.config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Define model configurations
        model_configs = [
            ("time", neural_data["time_labels"]),
            ("class", neural_data["class_labels"]),
            ("time_class", neural_data["combined_labels"]),
            ("shuffle_time", np.random.permutation(neural_data["time_labels"])),
            ("shuffle_class", np.random.permutation(neural_data["class_labels"])),
            (
                "shuffle_time_class",
                np.random.permutation(neural_data["combined_labels"]),
            ),
        ]

        for model_name, labels in model_configs:
            self.logger.info(f"Training {model_name} model...")
            model = self.create_cebra_model()
            model.fit(neural_data["neural_data"], labels)
            models[model_name] = model

            # Save model
            model_path = models_dir / f"cebra_{model_name}_model.pt"
            model.save(str(model_path))
            self.logger.info(f"Saved model: {model_path}")

        return models

    def load_models(self, model_names: list[str]) -> dict[str, CEBRA]:
        """Load trained models from disk."""
        models = {}
        models_dir = Path(self.config.MODELS_DIR)

        for name in model_names:
            model_path = models_dir / f"cebra_{name}_model.pt"
            try:
                models[name] = CEBRA.load(str(model_path))
                self.logger.info(f"Loaded model: {model_path}")
            except FileNotFoundError:
                self.logger.warning(f"Model file not found: {model_path}")
        return models

    def generate_embeddings(
        self, models: dict[str, CEBRA], neural_data: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Generate embeddings for all models."""
        embeddings = {}
        for name, model in models.items():
            self.logger.info(f"Generating embeddings for {name} model...")
            embeddings[name] = model.transform(neural_data)
        return embeddings


class Visualizer:
    """Handles all visualization tasks."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def plot_neural_data_overview(self, neural_data: dict) -> None:
        """Plot overview of neural data and class labels."""
        # Create figures directory
        figures_dir = Path(self.config.FIGURES_DIR)
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(9, 3), dpi=self.config.FIGURE_DPI)
        plt.subplots_adjust(wspace=0.5)

        # Plot neural data
        ax1 = plt.subplot(121)
        ax1.imshow(neural_data["neural_data"].T, aspect="auto", cmap="gray_r")
        ax1.set_ylabel("Channels #")
        ax1.set_xlabel("Time [s]")
        ax1.set_xticks(
            np.linspace(0, neural_data["neural_data"].shape[0], 5),
            np.linspace(0, neural_data["neural_data"].shape[0] / 250, 5, dtype=int),
        )

        # Plot class labels
        ax2 = plt.subplot(122)
        ax2.scatter(
            np.arange(len(neural_data["class_labels"])),
            neural_data["class_labels"],
            c="gray",
            s=1,
        )
        ax2.set_ylabel("Class #")
        ax2.set_xlabel("Time [s]")
        ax2.set_xticks(
            np.linspace(0, neural_data["neural_data"].shape[0], 5),
            np.linspace(0, neural_data["neural_data"].shape[0] / 250, 5, dtype=int),
        )
        ax2.set_yticks(
            np.arange(len(neural_data["class_names"])), neural_data["class_names"]
        )

        plt.tight_layout()
        save_path = figures_dir / "neural_data_overview.svg"
        fig.savefig(
            save_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight", transparent=True
        )
        plt.show()
        self.logger.info(f"Saved neural data overview to {save_path}")

    def plot_embeddings_comparison(self, embeddings: dict, neural_data: dict) -> None:
        """Plot comparison of different CEBRA embeddings."""
        figures_dir = Path(self.config.FIGURES_DIR)
        figures_dir.mkdir(parents=True, exist_ok=True)

        before_onset = neural_data["combined_labels"][:, 0] < 0
        after_onset = neural_data["combined_labels"][:, 0] >= 0

        fig = plt.figure(figsize=(12, 8), dpi=self.config.FIGURE_DPI)
        gs = GridSpec(2, 3, figure=fig, wspace=-0.3, hspace=-0.1)

        embedding_names = [
            "time",
            "class",
            "time_class",
            "shuffle_time",
            "shuffle_class",
            "shuffle_time_class",
        ]
        titles = [
            "Time",
            "Class",
            "Time+Class",
            "Time, Shuffled",
            "Class, Shuffled",
            "Time+Class, Shuffled",
        ]

        axes = [
            fig.add_subplot(gs[i, j], projection="3d")
            for i in range(2)
            for j in range(3)
        ]

        for ax, embedding_name, title in zip(axes, embedding_names, titles):
            if embedding_name in embeddings:
                embedding = embeddings[embedding_name]
                for process, cmap in zip(
                    [before_onset, after_onset], [self.config.CMAP1, self.config.CMAP2]
                ):
                    plot_embedding(
                        ax=ax,
                        embedding_labels=neural_data["combined_labels"][process][:, 0],
                        embedding=embedding[process],
                        cmap=cmap,
                        alpha=0.8,
                        markersize=1,
                        dpi=self.config.FIGURE_DPI,
                    )
                ax.set_title(title, fontsize=12, pad=-1)
                ax.grid(False)
                ax.set_axis_off()

        self._add_colorbar(fig)
        # plt.tight_layout()
        save_path = figures_dir / "cebra_embeddings_comparison.svg"
        fig.savefig(
            save_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight", transparent=True
        )
        plt.show()
        self.logger.info(f"Saved embeddings comparison to {save_path}")

    def _add_colorbar(self, fig) -> None:
        """Add colorbar to the embedding comparison plot."""
        cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])

        cmap1_obj = getattr(plt.cm, self.config.CMAP1)
        cmap2_obj = getattr(plt.cm, self.config.CMAP2)

        combined_cmap = ListedColormap(
            np.vstack(
                (
                    cmap1_obj(np.linspace(0, 1, self.config.BASELINE)),
                    cmap2_obj(np.linspace(0, 1, self.config.STIM_ONSET)),
                )
            )
        )

        sm = plt.cm.ScalarMappable(cmap=combined_cmap)
        sm.set_array(np.linspace(-self.config.BASELINE, self.config.STIM_ONSET, 900))

        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([-self.config.BASELINE, 0, 400, self.config.STIM_ONSET])
        cbar.set_ticklabels([-0.1, "onset", 0.4, 0.8], fontsize=8)
        cbar.ax.set_title("time(s)", fontsize=10)

    def plot_training_loss(self, models: dict[str, CEBRA]) -> None:
        """Plot training loss for all models."""
        figures_dir = Path(self.config.FIGURES_DIR)
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(6, 3), dpi=self.config.FIGURE_DPI)
        ax = plt.subplot(111)

        model_configs = [
            ("shuffle_time_class", "gray", 0.3, "time + class, shuffled"),
            ("shuffle_class", "gray", 0.6, "class, shuffled"),
            ("shuffle_time", "gray", 1, "time, shuffled"),
            ("class", "blue", 0.3, "class"),
            ("time", "skyblue", 0.6, "time"),
            ("time_class", "deepskyblue", 0.6, "time + class"),
        ]

        for model_name, color, alpha, label in model_configs:
            if model_name in models:
                ax = cebra.plot_loss(
                    models[model_name],
                    color=color,
                    alpha=alpha,
                    label=label,
                    ax=ax,
                    linewidth=1,
                    dpi=self.config.FIGURE_DPI,
                )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("InfoNCE Loss")
        ax.set_xlim(0, self.config.MAX_ITERATIONS)
        plt.legend(bbox_to_anchor=(0.61, 0.57), loc="best")
        # plt.tight_layout()

        save_path = figures_dir / "cebra_training_loss.svg"
        fig.savefig(
            save_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight", transparent=True
        )
        plt.show()
        self.logger.info(f"Saved training loss plot to {save_path}")


# class AnimationGenerator:
#     """Generates animated visualizations of embeddings."""

#     def __init__(self, config: Config, logger: logging.Logger):
#         self.config = config
#         self.logger = logger

#     def create_rotating_gif(
#         self,
#         embedding: np.ndarray,
#         labels: np.ndarray,
#         title: str,
#         output_filename: str,
#     ) -> None:
#         """Create a rotating GIF of 3D embedding."""
#         animations_dir = Path(self.config.ANIMATIONS_DIR)
#         animations_dir.mkdir(parents=True, exist_ok=True)

#         def create_frame(angle: int) -> np.ndarray:
#             camera = dict(
#                 eye=dict(
#                     x=2 * np.cos(np.radians(angle)),
#                     y=2 * np.sin(np.radians(angle)),
#                     z=1,
#                 )
#             )

#             fig = px.line_3d(
#                 x=embedding[:, 0],
#                 y=embedding[:, 1],
#                 z=embedding[:, 2],
#                 color=labels,
#                 title=title,
#             )

#             fig.update_layout(scene_camera=camera)
#             fig.update_traces(marker=dict(size=2), showlegend=False)
#             fig.update_layout(
#                 scene=dict(
#                     xaxis=dict(visible=False),
#                     yaxis=dict(visible=False),
#                     zaxis=dict(visible=False),
#                 )
#             )

#             img_bytes = fig.to_image(format="png", scale=2)
#             return imageio.imread(img_bytes)

#         angles = range(0, self.config.ANIMATION_FRAMES, 1)

#         with tqdm_joblib(desc=f"Generating {title} GIF frames", total=len(angles)):
#             frames = Parallel(n_jobs=-1)(
#                 delayed(create_frame)(angle) for angle in angles
#             )

#         output_path = animations_dir / output_filename
#         imageio.mimsave(output_path, frames, fps=self.config.ANIMATION_FPS, loop=0)
#         self.logger.info(f"GIF saved as {output_path}")


# %%
if __name__ == "__main__":
    """Main execution function."""
    config = Config()
    logger = setup_logging(config)
    evo_dir = Path("../HAD-MEEG_results/grand_evo")
    evo = load(evo_dir / "grand_evo_eeg.pkl")
    cls_names = evo.metadata["superclass_level1"].unique().tolist()
    cls_data = {}
    for cls_id in cls_names:
        e = evo[evo.metadata["superclass_level1"] == cls_id]
        if isinstance(e, mne.Evoked):
            grand_e = e
        else:
            grand_e = mne.grand_average(e)
        data = grand_e._data  # shape: (n_channels, n_times)
        cls_data[cls_id] = data

    neural_data = prepare_neural_data(
        balanced_evos=cls_data,
        time_samples=evo[0].times.size,
        time_start=evo[0].times[0],
        time_end=evo[0].times[-1],
    )
    # %%
    # Initialize components
    preprocessor = DataPreprocessor(config, logger)
    model_manager = CEBRAModelManager(config, logger)
    visualizer = Visualizer(config, logger)
    # animator = AnimationGenerator(config, logger)

    # Load and preprocess data
    logger.info("Starting CEBRA analysis pipeline...")
    preprocessor.save_processed_data(neural_data)

    # Visualize data overview
    visualizer.plot_neural_data_overview(neural_data)
    # %%
    # Train models or load existing ones
    model_names = [
        "time",
        "class",
        "time_class",
        "shuffle_time",
        "shuffle_class",
        "shuffle_time_class",
    ]

    try:
        models = model_manager.load_models(model_names)
        if len(models) < len(model_names):
            logger.info("Some models missing, training all models...")
            models = model_manager.train_models(neural_data)
    except Exception:
        logger.info("Training new models...")
        models = model_manager.train_models(neural_data)

    # Generate embeddings
    embeddings = model_manager.generate_embeddings(models, neural_data["neural_data"])

    # Create visualizations
    visualizer.plot_embeddings_comparison(embeddings, neural_data)
    visualizer.plot_training_loss(models)

    # Generate animations
    # id_map = {
    #     i: neural_data["class_names"][index]
    #     for index, i in enumerate(neural_data["numeric_labels"])
    # }
    # str_labels = np.array([id_map[i] for i in neural_data["class_labels"]])

    # for embedding_name in ["time", "class", "time_class"]:
    #     if embedding_name in embeddings:
    #         animator.create_rotating_gif(
    #             embeddings[embedding_name],
    #             str_labels,
    #             f"CEBRA-{embedding_name.replace('_', '+')}",
    #             f"cebra_{embedding_name}.gif",
    #         )

    logger.info("CEBRA analysis pipeline completed successfully!")
# %%

# time_cls_model = CEBRA.load(
#     "../HAD-MEEG_results/cebra/models/cebra_time_class_model.pt"
# )
# train_embedding = time_cls_model.transform(neural_data["numeric_labels"])
# train_continuous_label = neural_data["combined_labels"]
# fig = cebra.integrations.plotly.plot_embedding_interactive(
#     train_embedding,
#     embedding_labels=train_continuous_label[:, 0],
#     title="Time+class CEBRA Embedding",
#     markersize=2,
#     cmap="magma_r",
# )
# %%
