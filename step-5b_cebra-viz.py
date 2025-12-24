# %%
import numpy as np
from joblib import load
import matplotlib as mpl
import umap
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import colors as mcolors
from pathlib import Path
from src import DataConfig
from src.cebra import (
    plot_cebra_label_meeg,
    make_point_colors,
    ModelManager,
    _norm,
)
from tqdm.auto import tqdm

# %%
cfg = DataConfig()
SAVE_DIR = cfg.results_root / "cebra"
MODEL_DIR = SAVE_DIR / "models"
FIG_DIR = SAVE_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FONT_SIZE = 12
FONT_PATH = Path("resources") / "Helvetica.ttc"
DEVICE = "cuda:0"
RANDOM_SEED = 42

fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
data = load(SAVE_DIR / "cebra_input_meeg.pkl")
meg = data["meg"]
eeg = data["eeg"]
labels = data["plot_labels"]
cls_color = {
    "Sports": "#0ba869",
    "Personal": "#df978d",
    "Household": "#cfc050",
    "Socializing": "#21a5c0",
    "Eating": "#73b539",
    "*baseline": "lightgray",
}


def umap_it(
    X: np.ndarray,
    n_components: int = 2,
    min_dist: float = 0.005,
    n_neighbors: int = 100,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Perform UMAP dimensionality reduction."""
    X_scaled = _norm(X)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    Z = reducer.fit_transform(X_scaled)
    return Z


def c2cmap(end_color, start_color="lightgray", gmma=1.5) -> "cmap":
    return mcolors.LinearSegmentedColormap.from_list(
        "it_cmap",
        [start_color, end_color],
        N=256,
        gamma=gmma,
    )


# %%
meg_models = ModelManager(
    MODEL_DIR / "meg", _norm(meg["X"]), meg["Y"], labels.get("*baseline", None)
)
eeg_models = ModelManager(
    MODEL_DIR / "eeg", _norm(eeg["X"]), eeg["Y"], labels.get("*baseline", None)
)
meg_embs = meg_models.transform_all()
eeg_embs = eeg_models.transform_all()
y = meg_models.rm_label_y
# %% color palette
time_cmap = c2cmap("gray", "lightgray", gmma=1.0)
cls_cmap = {labels[label]: c2cmap(color) for label, color in cls_color.items()}
sample_colors = make_point_colors(meg_models.rm_label_y, cls_cmap)
# %%
fig_train = plot_cebra_label_meeg(
    X_eeg=eeg_models.rm_label_X,
    Y_eeg=y,
    X_meg=meg_models.rm_label_X,
    Y_meg=y,
    cls_label={k: v for k, v in labels.items() if k != "*baseline"},
    figsize=(6, 4.5),
    cmap_time=time_cmap,
    cmap_signal="Greys",
)[0]
fig_train.savefig(
    SAVE_DIR / "figs" / "meeg_cebra_train_data.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
colorbar_fig, colorbar_ax = plt.subplots(5, 1, figsize=(2, 1.5))
cls_cmap_msg = {k: cls_cmap[v] for k, v in labels.items() if k != "*baseline"}
clss = list(cls_cmap_msg.keys())
for i, ax in enumerate(colorbar_ax):
    ax.axis("off")
    cmap = cls_cmap_msg[clss[i]]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
    )
    ax.text(0, 0.5, clss[i], ha="left", va="center", fontsize=FONT_SIZE, color="black")
colorbar_fig.savefig(
    SAVE_DIR / "figs" / "meeg_cebra_colorbar.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
train_para = "it-10000_wb-False"
meg_embs_show = {
    "class_only": meg_embs[train_para]["class_only"],
    "class_time": meg_embs[train_para]["class_time"],
    "class_only_shuffle": meg_embs[train_para]["class_only_shuffle"],
    "class_time_shuffle": meg_embs[train_para]["class_time_shuffle"],
}
eeg_embs_show = {
    "class_only": eeg_embs[train_para]["class_only"],
    "class_time": eeg_embs[train_para]["class_time"],
    "class_only_shuffle": eeg_embs[train_para]["class_only_shuffle"],
    "class_time_shuffle": eeg_embs[train_para]["class_time_shuffle"],
}

fig_3d_cebra, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6),
    subplot_kw={"projection": "3d"},
)
axes_dict = {
    (0, 0): "class",
    (0, 1): "class+time",
    (0, 2): "class",
    (0, 3): "class+time",
    (1, 0): "class_shuffle",
    (1, 1): "class+time_shuffle",
    (1, 2): "class_shuffle",
    (1, 3): "class+time_shuffle",
}
for (i, j), msg in axes_dict.items():
    ax = axes[i, j]
    ax.set_title(msg, fontsize=FONT_SIZE)
    ax.axis("off")
    match j:
        case 0 | 1:
            embs = meg_embs_show
        case 2 | 3:
            embs = eeg_embs_show
    match msg:
        case "class":
            emb = embs["class_only"]
        case "class+time":
            emb = embs["class_time"]
        case "class_shuffle":
            emb = embs["class_only_shuffle"]
        case "class+time_shuffle":
            emb = embs["class_time_shuffle"]

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        emb[:, 2],
        c=sample_colors,
        s=0.1,
        alpha=1,
        rasterized=True,
    )
    ax.dist = 10
plt.subplots_adjust(wspace=0, hspace=0)
fig_3d_cebra.savefig(
    SAVE_DIR / "figs" / "meeg_3d_embeddings.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
meg_umap = {
    key: umap_it(emb, n_components=2, min_dist=0.7, n_neighbors=100)
    for key, emb in tqdm(meg_embs_show.items(), desc="umap meg")
}
eeg_umap = {
    key: umap_it(emb, n_components=2, min_dist=0.7, n_neighbors=100)
    for key, emb in tqdm(eeg_embs_show.items(), desc="umap eeg")
}
# %%
fig_3d_umap = plt.figure(figsize=(12, 6))
axes = fig_3d_umap.subplots(2, 4)
axes_dict = {
    (0, 0): "class",
    (0, 1): "class+time",
    (0, 2): "class",
    (0, 3): "class+time",
    (1, 0): "class_shuffle",
    (1, 1): "class+time_shuffle",
    (1, 2): "class_shuffle",
    (1, 3): "class+time_shuffle",
}
for (i, j), msg in axes_dict.items():
    ax = axes[i, j]
    ax.set_title(msg, fontsize=FONT_SIZE)
    ax.axis("off")
    match j:
        case 0 | 1:
            embs = meg_umap
        case 2 | 3:
            embs = eeg_umap
    match msg:
        case "class":
            emb = embs["class_only"]
        case "class+time":
            emb = embs["class_time"]
        case "class_shuffle":
            emb = embs["class_only_shuffle"]
        case "class+time_shuffle":
            emb = embs["class_time_shuffle"]

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=sample_colors,
        s=0.5,
        alpha=1,
        rasterized=True,
    )
# plt.subplots_adjust(wspace=0, hspace=0)
fig_3d_umap.savefig(
    SAVE_DIR / "figs" / "meeg_2d_umap_embeddings.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
fig_3d_cebra_umap = plt.figure(figsize=(12, 6))
axes = fig_3d_cebra_umap.subplots(2, 4)

axes_dict = {
    (0, 0): "class",
    (0, 1): "class+time",
    (0, 2): "class",
    (0, 3): "class+time",
    (1, 0): "class",
    (1, 1): "class+time",
    (1, 2): "class",
    (1, 3): "class+time",
}

for (i, j), msg in axes_dict.items():
    ax = axes[i, j]
    ax.set_title(msg, fontsize=FONT_SIZE)
    ax.axis("off")

    # 选 embs
    match (i, j):
        case (0, 0) | (0, 1):
            embs = meg_embs_show
        case (0, 2) | (0, 3):
            embs = eeg_embs_show
        case (1, 0) | (1, 1):
            embs = meg_umap
        case (1, 2) | (1, 3):
            embs = eeg_umap

    match msg:
        case "class":
            emb = embs["class_only"]
        case "class+time":
            emb = embs["class_time"]
        case "class_shuffle":
            emb = embs["class_only_shuffle"]
        case "class+time_shuffle":
            emb = embs["class_time_shuffle"]

    ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=sample_colors,
        s=0.5,
        alpha=0.5,
        rasterized=True,
    )

fig_3d_cebra_umap.savefig(
    SAVE_DIR / "figs" / "meeg_cebra_umap_embeddings.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
fig_cbar = plt.subplots(5, 1, figsize=(2, 1.5))
cls_cmap_msg = {k: cls_cmap[v] for k, v in labels.items() if k != "*baseline"}
clss = list(cls_cmap_msg.keys())
for i, ax in enumerate(colorbar_ax):
    ax.axis("off")
    cmap = cls_cmap_msg[clss[i]]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
    )
    # ax.text(0, 0.5, clss[i], ha='left', va='center', fontsize=FONT_SIZE, color='black')
# %%
train_para = "it-10000_wb-False"

meg_models_to_show = {
    "class_only": meg_models.models[train_para]["class_only"],
    "class_time": meg_models.models[train_para]["class_time"],
    "class_only_shuffle": meg_models.models[train_para]["class_only_shuffle"],
    "class_time_shuffle": meg_models.models[train_para]["class_time_shuffle"],
}
eeg_models_to_show = {
    "class_only": eeg_models.models[train_para]["class_only"],
    "class_time": eeg_models.models[train_para]["class_time"],
    "class_only_shuffle": eeg_models.models[train_para]["class_only_shuffle"],
    "class_time_shuffle": eeg_models.models[train_para]["class_time_shuffle"],
}
# %%
fig_loss, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True, dpi=300)
axes[0].plot(
    meg_models_to_show["class_only_shuffle"].state_dict_["loss"],
    c="lightgray",
    label="class shuffled",
)
axes[0].plot(
    meg_models_to_show["class_time_shuffle"].state_dict_["loss"],
    c="gray",
    label="class+time shuffled",
)
axes[0].plot(
    meg_models_to_show["class_only"].state_dict_["loss"],
    c="skyblue",
    label="class",
    alpha=0.5,
)
axes[0].plot(
    meg_models_to_show["class_time"].state_dict_["loss"],
    c="deepskyblue",
    label="class+time",
)

axes[1].plot(
    eeg_models_to_show["class_only_shuffle"].state_dict_["loss"],
    c="lightgray",
    label="class shuffled",
)
axes[1].plot(
    eeg_models_to_show["class_time_shuffle"].state_dict_["loss"],
    c="gray",
    label="class+time shuffled",
)
axes[1].plot(
    eeg_models_to_show["class_only"].state_dict_["loss"],
    c="skyblue",
    label="class",
    alpha=0.5,
)
axes[1].plot(
    eeg_models_to_show["class_time"].state_dict_["loss"],
    c="deepskyblue",
    label="class+time",
)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].spines["bottom"].set_visible(False)
axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].set_xlabel("Iterations", fontsize=FONT_SIZE)
axes[0].set_ylabel("MEG InfoNCE Loss", fontsize=FONT_SIZE)
axes[1].set_ylabel("EEG InfoNCE Loss", fontsize=FONT_SIZE)

plt.legend(
    ncol=2,
    handlelength=1,
    columnspacing=2,
    handletextpad=0.3,
    handleheight=0,
    fontsize=FONT_SIZE - 2,
    loc="center left",
    bbox_to_anchor=(0.05, 1.2),
    # 取消legend的边框
    frameon=False,
)
plt.show()

fig_loss.savefig(
    SAVE_DIR / "figs" / "meeg_cebra_loss.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%

# %%
