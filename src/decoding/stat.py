from scipy.stats import ttest_1samp
import numpy as np


def sliding_1sample_ttest(
    scores: np.ndarray,
    n_timepoints: int,
    baseline: float = 0.5,
    alpha: float = 0.01,
    tail: str = "greater",
):
    """
    Perform a differential test using single-sample t-test for each time point.

    Parameters
    ----------
    scores : ndarray
        Array of shape (n_samples, n_timepoints) containing the scores.
    n_timepoints : int
        Number of time points.
    baseline : float, optional
        Baseline value for the t-test. Defaults to 0.5.
    alpha : float, optional
        Significance level for the t-test. Defaults to 0.01.
    tail : str, optional
        Type of t-test ('two-sided', 'greater', 'less'). Defaults to 'greater'.

    Returns
    -------
    tuple
        A tuple containing the t-values, p-values, and indices of significant time points.
    """
    t_values = []
    p_values = []
    for i in range(n_timepoints):
        t, p = ttest_1samp(scores[:, i], baseline)
        if tail == "greater":
            p = p / 2 if t > 0 else 1 - p / 2
        elif tail == "less":
            p = p / 2 if t < 0 else 1 - p / 2

        t_values.append(t)
        p_values.append(p)

    t_values = np.array(t_values)
    p_values = np.array(p_values)

    # Bonferroni correction
    corrected_alpha = alpha / n_timepoints
    significant_timepoints = np.where(p_values < corrected_alpha)[0]

    return t_values, p_values, significant_timepoints
