# %%
import numpy as np
from scipy.spatial.distance import cdist


# %%
class TimeRDM:
    def __init__(self, data: np.ndarray | list[np.ndarray]) -> None:
        """Initialize the TimeRDM class with MNE Epochs.

        Parameters
        ----------
        data : np.ndarray
            The data array, shoule be (n_conditions, n_features, n_times).
        """
        self.data = data

    @staticmethod
    def compute(
        data: np.ndarray,
        fill_diagonal: float = np.nan,
        metric: str = "pearson",
        normalize: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        """Compute Representational Dissimilarity Matrices (RDMs) for each condition.

        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_conditions, n_features, n_times).
        fill_diagonal : float | np.nan, optional
            Value to fill the diagonal of the RDMs, by default np.nan.
        metric : str, optional
            Distance metric to use, by default 'correlation'.

        Returns
        -------
        np.ndarray or list[np.ndarray]
            Spatiotemporal RDM if `is_spatiotemporal` is True, else a list of RDMs for each time point.
        """

        # Compute RDM for each time point
        rdms = []
        n_times = data.shape[2]
        for t_idx in range(n_times):
            data_matrix = data[:, :, t_idx]  # n_conditions x n_features
            if normalize:
                data_matrix = TimeRDM._zscore_data(data_matrix)
            # Compute RDM using specified metric
            if metric == "pearson":
                rdm = -np.corrcoef(data_matrix)
            else:
                rdm = cdist(data_matrix, data_matrix, metric=metric)
            np.fill_diagonal(rdm, fill_diagonal)
            rdms.append(rdm)
        return np.array(rdms)  # n_times x n_conditions x n_conditions

    @staticmethod
    def _zscore_data(data: np.ndarray) -> np.ndarray:
        """Z-score the data across the first axis.

        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_conditions, n_features).

        Returns
        -------
        np.ndarray
            Z-scored data.
        """
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, ddof=1, keepdims=True)
        zscored = (data - mean) / std
        return zscored


# %%
