import numpy as np
import cebra
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

__all__ = ["get_model", "ModelZoo", "_norm", "ModelManager"]


def get_model(
    inpout_type: str,
    model_receptive_field: int = 36,  # in samples, 0.18s for 200Hz
    time_delta: bool = False,
    offset: int = 20,
    output_dimension: int = 3,
    delta: float = 20,  # 200Hz -> 10 samples = 0.05s
    device: str = "cuda:0",
    batch_size: int = 1024,
    verbose: bool = True,
    max_iterations: int = 50_000,
    num_hidden_units: int = 64,
) -> cebra.CEBRA:
    """Create a CEBRA model with specified parameters.

    This is a convenience wrapper around trial_cebra.
    """
    # Import here to avoid circular dependency
    from . import trial_cebra

    return trial_cebra(
        input_type=inpout_type,
        model_receptive_field=model_receptive_field,
        output_dimension=output_dimension,
        delta=delta,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
        max_iterations=max_iterations,
        num_hidden_units=num_hidden_units,
        time_delta=time_delta,
        offset=offset,
    )


def _norm(x: np.ndarray, return_scaler: bool = False) -> np.ndarray:
    scaler = StandardScaler()
    if return_scaler:
        return scaler.fit_transform(x), scaler
    return scaler.fit_transform(x)


class ModelZoo:
    """智能模型管理器：自动追踪训练状态，按需加载，训练后自动保存

    Features:
    - 延迟加载：模型仅在需要时才加载到内存
    - 自动保存：模型训练后立即保存
    - 状态追踪：追踪每个模型的训练状态
    - 智能跳过：已训练的模型自动跳过，避免重复训练

    Parameters
    ----------
    X : np.ndarray
        输入数据，shape (n_samples, n_features, n_timepoints)
    Y : np.ndarray
        标签数据，shape (n_samples, 2)，[:, 0] 为时间，[:, 1] 为类别
    model_types : list[str]
        模型类型列表，可选 ['time_only', 'class_only', 'class_time']
    rm_label : int, optional
        要移除的标签值（通常用于移除 baseline）
    save_dir : Path
        模型保存目录
    *args, **kwargs
        传递给 get_model 的其他参数

    Examples
    --------
    >>> # 首次运行：训练并自动保存
    >>> models = ModelZoo(X=data, Y=labels, model_types=['class_time'])
    >>> models.fit_all(with_shuffle=True)

    >>> # 再次运行：自动加载已训练模型，跳过训练
    >>> models = ModelZoo(X=data, Y=labels, model_types=['class_time'])
    >>> models.fit_all(with_shuffle=True)  # 输出: "⊙ Model already fitted, skipping"

    >>> # 强制重新训练
    >>> models.fit_all(with_shuffle=True, force_refit=True)

    >>> # 查看训练状态
    >>> print(models.get_status())
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        model_types: list[str] = ["time_only", "class_only", "class_time"],
        rm_label: int = None,
        save_dir: Path = Path("results/cebra/models"),
        random_seed: int = 42,
        *args,
        **kwargs,
    ):
        self.models = {}
        self.shuffle_models = {}
        self.fitted_status = {}  # 追踪每个模型的训练状态
        self.shuffle_fitted_status = {}
        self.random_seed = random_seed

        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.model_types = model_types
        self.model_kwargs = {"args": args, "kwargs": kwargs}

        # 初始化训练状态
        for model_type in model_types:
            self.fitted_status[model_type] = False
            self.shuffle_fitted_status[model_type] = False

        # 处理数据：移除指定标签
        if rm_label is not None:
            rm_indices = np.where(Y[:, 1] == rm_label)[0]
            X = np.delete(X, rm_indices, axis=0)
            Y = np.delete(Y, rm_indices, axis=0)

        self.X = X
        self.Y = Y
        self.y_class = Y[:, 1].astype(np.int32)
        self.y_time = Y[:, 0]

    def _get_model_path(self, model_type: str, shuffle: bool = False) -> Path:
        """获取模型文件路径"""
        suffix = "_shuffle" if shuffle else ""
        return self.save_dir / f"{model_type}{suffix}.pt"

    def _model_exists(self, model_type: str, shuffle: bool = False) -> bool:
        """检查模型文件是否存在"""
        return self._get_model_path(model_type, shuffle).exists()

    def _ensure_model_loaded(self, model_type: str, shuffle: bool = False) -> None:
        """确保模型已加载（延迟加载策略）"""
        model_dict = self.shuffle_models if shuffle else self.models
        status_dict = self.shuffle_fitted_status if shuffle else self.fitted_status

        # 如果模型已在内存中，直接返回
        if model_type in model_dict:
            return

        # 尝试从磁盘加载已训练的模型
        if self._model_exists(model_type, shuffle):
            try:
                model_dict[model_type] = cebra.CEBRA.load(
                    self._get_model_path(model_type, shuffle), weights_only=False
                )
                status_dict[model_type] = True
                logger.success(
                    f"✓ Loaded {'shuffle ' if shuffle else ''}model: {model_type}"
                )
            except Exception as e:
                logger.warning(f"Failed to load {model_type}, will create new: {e}")
                model_dict[model_type] = get_model(
                    inpout_type=model_type,
                    *self.model_kwargs["args"],
                    **self.model_kwargs["kwargs"],
                )
                status_dict[model_type] = False
        else:
            # 创建新模型
            model_dict[model_type] = get_model(
                inpout_type=model_type,
                *self.model_kwargs["args"],
                **self.model_kwargs["kwargs"],
            )
            status_dict[model_type] = False

    def fit(
        self, model_type: str, on_shuffle: bool = False, force_refit: bool = False
    ) -> None:
        """训练模型，训练后自动保存

        Args:
            model_type: 模型类型
            on_shuffle: 是否使用打乱的标签
            force_refit: 是否强制重新训练（忽略已有模型）
        """
        # 先尝试加载模型（如果磁盘上存在已训练的模型，会更新状态）
        self._ensure_model_loaded(model_type, on_shuffle)

        status_dict = self.shuffle_fitted_status if on_shuffle else self.fitted_status

        # 智能跳过：如果已训练且不强制重训，则跳过
        if status_dict.get(model_type, False) and not force_refit:
            logger.info(
                f"⊙ {'Shuffle ' if on_shuffle else ''}Model {model_type} already fitted, skipping"
            )
            return

        model_dict = self.shuffle_models if on_shuffle else self.models
        model = model_dict[model_type]

        # 准备训练数据
        if on_shuffle:
            y = self._shuffle_labels(self.Y)
            y_time = y[:, 0]
            y_class = y[:, 1].astype(np.int32)
        else:
            y_time = self.y_time
            y_class = self.y_class

        # 训练模型
        logger.info(f"→ Fitting {'shuffle ' if on_shuffle else ''}model: {model_type}")
        match model_type:
            case "time_only":
                model.fit(self.X, y_time)
            case "class_only":
                model.fit(self.X, y_class)
            case "class_time":
                model.fit(self.X, y_time, y_class)

        # 更新状态并自动保存
        status_dict[model_type] = True
        model.save(self._get_model_path(model_type, on_shuffle))
        logger.success(f"✓ Saved {'shuffle ' if on_shuffle else ''}model: {model_type}")

    def transform(self, model_type: str, on_shuffle: bool = False) -> np.ndarray:
        """转换数据为嵌入空间"""
        # 确保模型已加载
        self._ensure_model_loaded(model_type, on_shuffle)

        model_dict = self.shuffle_models if on_shuffle else self.models
        status_dict = self.shuffle_fitted_status if on_shuffle else self.fitted_status

        # 检查模型是否已训练
        if not status_dict.get(model_type, False):
            raise ValueError(
                f"{'Shuffle ' if on_shuffle else ''}Model {model_type} has not been fitted yet. "
                f"Call fit() first."
            )

        return model_dict[model_type].transform(self.X)

    def fit_all(self, with_shuffle: bool = False, force_refit: bool = False) -> None:
        """训练所有模型，自动跳过已训练的模型

        Args:
            with_shuffle: 是否同时训练打乱标签的模型
            force_refit: 是否强制重新训练所有模型
        """
        total_models = len(self.model_types) * (2 if with_shuffle else 1)
        logger.info(f"→ Fitting up to {total_models} models (auto-skip trained)")

        for model_type in self.model_types:
            self.fit(model_type, on_shuffle=False, force_refit=force_refit)

        if with_shuffle:
            for model_type in self.model_types:
                self.fit(model_type, on_shuffle=True, force_refit=force_refit)

    def transform_all(self, with_shuffle: bool = False) -> dict[str, np.ndarray]:
        """转换所有模型的数据"""
        total = len(self.model_types) * (2 if with_shuffle else 1)
        logger.info(f"→ Transforming {total} embeddings")

        embeddings = {}
        for model_type in self.model_types:
            embeddings[model_type] = self.transform(model_type, on_shuffle=False)

        if with_shuffle:
            for model_type in self.model_types:
                embeddings[f"{model_type}_shuffle"] = self.transform(
                    model_type, on_shuffle=True
                )

        return embeddings

    def get_status(self) -> dict:
        """获取所有模型的训练状态"""
        return {
            "normal": self.fitted_status.copy(),
            "shuffle": self.shuffle_fitted_status.copy(),
        }

    def _shuffle_labels(self, y: np.ndarray) -> np.ndarray:
        """打乱标签（用于对照实验）"""
        rng = np.random.default_rng(self.random_seed)
        shuffled_indices = rng.permutation(len(y))
        return y[shuffled_indices]


class ModelManager:
    def __init__(
        self,
        save_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        rm_label: int | None = None,
    ):
        Model_dirs = list(save_dir.iterdir())
        models = {}
        for d in Model_dirs:
            if d.is_dir():
                train_name = d.name.split(".")[0]
                models[train_name] = {}

                model_files = list(d.iterdir())
                for p in model_files:
                    if p.suffix == ".pt":
                        model_name = p.stem
                        models[train_name][model_name] = cebra.CEBRA.load(
                            p, weights_only=False
                        )
        self.models = models
        self.X = X
        self.y = y
        self.rm_label_X, self.rm_label_y = self.rm_label_from_Xy(X, y, rm_label)
        self.rm_label = rm_label

    def transform_all(self) -> dict[str, dict[str, np.ndarray]]:
        embs = {}
        for train_name, models in self.models.items():
            embs[train_name] = {}
            if "wb-False" in train_name:
                X = self.rm_label_X
            else:
                X = self.X
            for model_name, model in models.items():
                embs[train_name][model_name] = model.transform(X)
        return embs

    def rm_label_from_Xy(
        self, X: np.ndarray, Y: np.ndarray, rm_label: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove samples with a specific label from X and Y."""
        rm_indices = np.where(Y[:, 1] == rm_label)[0]
        if len(rm_indices) > 0:
            X_new = np.delete(X, rm_indices, axis=0)
            Y_new = np.delete(Y, rm_indices, axis=0)
            return X_new, Y_new
        return X, Y
