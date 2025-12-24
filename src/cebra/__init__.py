from .prep import split_cebra_dataset as split
from .viz import (
    plot_cebra_label_meeg,
    plot_cebra_label,
    plot_embedding,
    make_point_colors,
)
from .models import ModelZoo, get_model, ModelManager, _norm
import torch
from typing import Literal
import cebra

__all__ = [
    "split",
    "plot_cebra_label_meeg",
    "plot_cebra_label",
    "trial_cebra",
    "plot_embedding",
    "make_point_colors",
    "ModelZoo",
    "get_model",
    "ModelManager",
    "_norm",
]


def trial_cebra(
    input_type: Literal["class_only", "class_time", "time_only"],
    time_delta: bool = False,
    offset: int = 10,
    model_receptive_field: int = 10,
    drop_out: bool = False,
    drop_out_more: bool = False,
    distance: str = "cosine",
    output_dimension: int = 3,
    delta: float | None = None,
    device: str | torch.device = "cuda_if_available",
    min_temperature: float = 0.1,
    max_iterations: int = 50_000,
    num_hidden_units: int = 32,
    batch_size: int = 512,
    verbose: bool = True,
) -> cebra.CEBRA:
    """
    Initialize a CEBRA model for epoched/evoked data without global continuous time.

    Design:
    - The index time of X is treated as meaningless (just concatenation order).
      Therefore, we never use hybrid mode.
    - Trial's time and other behavioral variables are treated as continuous conditions
      and passed explicitly via y.
    - For time-related tasks,if `conditional='delta'`, `delta` defines
      the neighborhood size in the label space. If `time_delta` is True,
      cebra will use the offset in X's time index define a window, and
      form a distribution over that window for positive sampling.

    Parameters
    ----------
    input_type : {'class_only', 'class_time', 'time_only'}
        - 'class_only' : only uses a discrete class label
        - 'time_only'  : only uses continuous time label(s)
        - 'class_time' : uses both continuous time and discrete class
    offset : int
        Offset used in the time index to define a window for positive sampling when time_delta is True.
    time_delta : bool
        Whether to use the offset in X's time index to define a window for positive sampling.
    model_receptive_field : int
        The receptive field of the CNN encoder, used to choose an offsetX-model.
        Only {1, 5, 10, 36} are supported here.
    drop_out : bool
        Whether to use the dropout variant of the encoder.
        Only supported for receptive field 36.
    drop_out_more : bool
        Whether to use the "more-dropout" variant of the encoder.
        Only supported for receptive field 36.
    distance : {'cosine', 'euclidean'}
        Distance used in the CEBRA loss.
        'euclidean' is only supported for receptive field 1 and 5 and uses *-mse models.
    output_dimension : int
        Dimensionality of the embedding space.
    delta : float | None
        Neighborhood size in the continuous label space, for time-related tasks.
        Units must match the units of the continuous labels (e.g. seconds).
        Required for 'time_only' and 'class_time'; ignored for 'class_only'.
    device : str | torch.device
        Device to run the model on ('cuda_if_available', 'cpu', 'cuda:0', etc.).
    min_temperature : float
        Lower bound for the temperature for we use `temperature_mode='auto'` here.
    max_iterations : int
        Maximum number of training iterations.
    num_hidden_units : int
        Number of hidden units in the encoder network.
    batch_size : int
        Batch size for training.

    Returns
    -------
    model : cebra.CEBRA
        The initialized CEBRA model.
    """
    predefined_types = ["class_only", "class_time", "time_only"]
    assert input_type in predefined_types, (
        f"input_type must be one of {predefined_types}"
    )

    valid_offsets = {1, 5, 10, 36}
    assert model_receptive_field in valid_offsets, (
        f"model_receptive_field must be one of {valid_offsets}"
    )

    if (drop_out or drop_out_more) and model_receptive_field != 36:
        raise ValueError("Dropout is only supported for receptive field 36.")
    if drop_out and drop_out_more:
        raise ValueError("drop_out and drop_out_more cannot both be True.")
    assert distance in {"euclidean", "cosine"}, (
        "distance must be 'euclidean' or 'cosine'"
    )

    # 1. 选模型结构
    if drop_out_more:
        model_arch = f"offset{model_receptive_field}-model-more-dropout"
    elif drop_out:
        model_arch = f"offset{model_receptive_field}-model-dropout"
    else:
        model_arch = f"offset{model_receptive_field}-model"
        if distance == "euclidean":
            if model_receptive_field not in {1, 5}:
                raise ValueError(
                    "Euclidean distance is only supported for receptive field 1 and 5."
                )
            model_arch += "-mse"

    # 2. 映射 input_type -> conditional / delta / time_offsets
    conditional: str | None = None
    delta_param: float | None = None
    time_offsets: int | None = None

    if input_type == "class_only":
        # 这里假定 class_only = 只用离散 class 做监督
        conditional = "delta"
        delta_param = 0.5 if delta is None else float(delta)

    elif input_type in {"time_only", "class_time"}:
        if time_delta:
            # 用 time_delta：offset 在 index 上定义时间窗，Δy 来自这个窗
            conditional = "time_delta"
            time_offsets = offset
            # 不需要 delta
        else:
            # 纯 delta：忽略 index 顺序，只在 y 空间用高斯球
            conditional = "delta"
            if delta is None:
                raise ValueError(
                    "For 'time_only'/'class_time' with time_delta=False, delta must be provided."
                )
            delta_param = float(delta)

    model = cebra.CEBRA(
        model_architecture=model_arch,
        device=device,
        conditional=conditional,
        delta=delta_param,
        distance=distance,
        temperature_mode="auto",
        min_temperature=min_temperature,
        max_iterations=max_iterations,
        output_dimension=output_dimension,
        num_hidden_units=num_hidden_units,
        hybrid=False,
        time_offsets=time_offsets if time_offsets is not None else 1,
        batch_size=batch_size,
        verbose=verbose,
    )
    return model
