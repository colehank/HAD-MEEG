# %%
import numpy as np
from scipy.stats import spearmanr, kendalltau
from numpy.typing import NDArray
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm.auto import tqdm


# %%
class RSA:
    """Perform RSA between RDMs.

    Parameters
    ----------
    rdm1 : NDArray
        The first RDM or a set of RDMs.
    rdm2 : NDArray
        The second RDM or a set of RDMs.
    input_type : str
        Type of input RDMs. One of 'n2n', 'one2n', 'one2one'.
    n_jobs : int
        The number of parallel jobs to run.
    n_iter : int
        The number of iterations for bootstrapping or permutation testing.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05.
        (only used when computing CI using bootstrap)
    """

    def __init__(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        input_type: str,
        n_jobs: int,
        n_iter: int,
        alpha: float = 0.05,
    ) -> None:
        self._check_input(rdm1, rdm2, input_type)

        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.input_type = input_type
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.alpha = alpha

    def compute(
        self,
        corr_method: str,
        sig_method: str,
    ) -> tuple:
        """
        Compute the correlation between RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        stats : NDArray
            The bootstrapped or permuted correlation coefficients,
            depending on `sig_method`.
            if `sig_method` is 'bootstrap',
            it is the bootstrapped confidence intervals(CI), or a list of CI.
            if `sig_method` is 'permutation',
            it is the permuted significance levels(p), or a list of p.
        """

        if self.input_type == "one2one":
            corr, sig = self.corr_rdm(corr_method, sig_method)

        elif self.input_type == "one2n":
            corr, sigs = self.corr_rdm_rdms(corr_method, sig_method)

        elif self.input_type == "n2n":
            corr, sigs = self.corr_rdms(corr_method, sig_method)
        else:
            raise ValueError('input_type must be one of "n2n", "one2n", "one2one"')

        if self.input_type == "one2one":
            return corr, sig
        else:
            return corr, sigs

    def corr_rdm(
        self,
        corr_method: str,
        sig_method: str,
    ) -> tuple[float, tuple] | tuple[float, float]:
        """
        Compute the correlation between two RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        stats : tuple or float
            The confidence interval (if `sig_method` is 'bootstrap')
            or p-value (if `sig_method` is 'permutation').
        """
        if sig_method == "bootstrap":
            corr, boot_cs = self.corr_bootstrap(
                self.rdm1,
                self.rdm2,
                corr_method,
                n_bootstraps=self.n_iter,
                n_jobs=self.n_jobs,
            )
            ci_lower = np.percentile(boot_cs, 100 * self.alpha / 2)
            ci_upper = np.percentile(boot_cs, 100 * (1 - self.alpha / 2))

            return corr, (ci_lower, ci_upper)

        elif sig_method == "permutation":
            corr, perm_cs = self.corr_permutation(
                self.rdm1,
                self.rdm2,
                corr_method,
                n_permutations=self.n_iter,
                n_jobs=self.n_jobs,
            )
            p = np.sum(perm_cs >= corr) / len(perm_cs)
            return corr, p
        else:
            raise ValueError('sig_method must be one of "bootstrap", "permutation"')

    def corr_rdms(
        self,
        corr_method: str,
        sig_method: str,
    ) -> tuple[list[float], list]:
        """
        Compute the correlations between pairs of RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.

        Returns
        -------
        corrs : list of float
            List of correlation coefficients between paired RDMs.
        ci_or_p : list
            List of confidence intervals (if `sig_method` is 'bootstrap')
            or p-values (if `sig_method` is 'permutation').
        """
        n_pair = self.rdm1.shape[0]
        corrs = []
        ci = []  # Initialize outside the loop
        ps = []  # Initialize outside the loop

        for i, (rdm1_i, rdm2_i) in tqdm(
            enumerate(zip(self.rdm1, self.rdm2)),
            desc="Computing paired RDM correlations:",
            total=n_pair,
        ):
            desc = f"{i + 1}/{n_pair}"

            if sig_method == "bootstrap":
                corr, boot_cs = self.corr_bootstrap(
                    rdm1_i,
                    rdm2_i,
                    corr_method,
                    n_bootstraps=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc,
                )
                ci_lower = np.percentile(boot_cs, 100 * self.alpha / 2)
                ci_upper = np.percentile(boot_cs, 100 * (1 - self.alpha / 2))
                ci.append((ci_lower, ci_upper))
                corrs.append(corr)

            elif sig_method == "permutation":
                corr, perm_cs = self.corr_permutation(
                    rdm1_i,
                    rdm2_i,
                    corr_method,
                    n_permutations=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc,
                )
                p = np.sum(perm_cs >= corr) / len(perm_cs)
                corrs.append(corr)
                ps.append(p)
            else:
                raise ValueError('sig_method must be one of "bootstrap", "permutation"')

        if sig_method == "bootstrap":
            return corrs, ci
        elif sig_method == "permutation":
            return corrs, ps

    def corr_rdm_rdms(
        self,
        corr_method: str,
        sig_method: str,
    ) -> tuple[list[float], list]:
        """
        Compute the correlations between a single RDM and a set of RDMs with statistical significance testing.

        Parameters
        ----------
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        sig_method : str
            The significance testing method. One of 'bootstrap', 'permutation'.

        Returns
        -------
        corrs : list of float
            List of correlation coefficients between the single RDM and each RDM in the set.
        ci_or_p : list
            List of confidence intervals (if `sig_method` is 'bootstrap') or p-values (if `sig_method` is 'permutation').
        """
        rdm1 = self.rdm1
        rdms = self.rdm2

        corrs = []
        ci = []  # Initialize outside the loop
        ps = []  # Initialize outside the loop

        for i, rdm in tqdm(
            enumerate(rdms), desc="Computing RDM to RDMs correlations:", total=len(rdms)
        ):
            desc = f"{i + 1}/{len(rdms)}"

            if sig_method == "bootstrap":
                corr, boot_cs = self.corr_bootstrap(
                    rdm1,
                    rdm,
                    corr_method,
                    n_bootstraps=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc,
                )
                ci_lower = np.percentile(boot_cs, 100 * self.alpha / 2)
                ci_upper = np.percentile(boot_cs, 100 * (1 - self.alpha / 2))
                corrs.append(corr)
                ci.append((ci_lower, ci_upper))

            elif sig_method == "permutation":
                corr, perm_cs = self.corr_permutation(
                    rdm1,
                    rdm,
                    corr_method,
                    n_permutations=self.n_iter,
                    n_jobs=self.n_jobs,
                    desc=desc,
                )
                p = np.sum(perm_cs >= corr) / len(perm_cs)
                corrs.append(corr)
                ps.append(p)
            else:
                raise ValueError('sig_method must be one of "bootstrap", "permutation"')

        if sig_method == "bootstrap":
            return corrs, ci
        elif sig_method == "permutation":
            return corrs, ps

    def corr_bootstrap(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        corr_method: str,
        n_bootstraps: int,
        n_jobs: int,
        **kwargs,
    ) -> tuple[float, NDArray]:
        """
        Compute the correlation between two RDMs and the bootstrapped confidence intervals.

        Parameters
        ----------
        rdm1 : NDArray
            The first RDM.
        rdm2 : NDArray
            The second RDM.
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        n_bootstraps : int
            The number of bootstrap samples.
        n_jobs : int
            The number of parallel jobs to run.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        boot_cs : NDArray
            The array of bootstrapped correlation coefficients.
        """
        v1 = self._rdm2vec(rdm1)
        v2 = self._rdm2vec(rdm2)
        corr = self._corr(v1, v2, corr_method)

        desc = kwargs.get("desc", "Bootstrap process:")
        with tqdm_joblib(total=n_bootstraps, desc=desc, leave=False, position=1):
            boot_cs = Parallel(n_jobs=n_jobs)(
                delayed(self._bootstrap)(v1, v2, corr_method)
                for _ in range(n_bootstraps)
            )

        return corr, np.array(boot_cs)

    def corr_permutation(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        corr_method: str,
        n_permutations: int,
        n_jobs: int,
        **kwargs,
    ) -> tuple[float, NDArray]:
        """
        Compute the correlation between two RDMs and the distribution of permuted correlations.

        Parameters
        ----------
        rdm1 : NDArray
            The first RDM.
        rdm2 : NDArray
            The second RDM.
        corr_method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.
        n_permutations : int
            The number of permutations.
        n_jobs : int
            The number of parallel jobs to run.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        corr : float
            The correlation coefficient between the two RDMs.
        perm_cs : NDArray
            The array of permuted correlation coefficients.
        """
        v1 = self._rdm2vec(rdm1)
        v2 = self._rdm2vec(rdm2)
        corr = self._corr(v1, v2, corr_method)

        desc = kwargs.get("desc", "Permutation process:")
        with tqdm_joblib(total=n_permutations, desc=desc, leave=False, position=1):
            perm_cs = Parallel(n_jobs=n_jobs)(
                delayed(self._permutation)(v1, v2, corr_method)
                for _ in range(n_permutations)
            )

        return corr, np.array(perm_cs)

    def _corr(
        self,
        v1: NDArray,
        v2: NDArray,
        method: str,
    ) -> float:
        """
        Compute the correlation between two vectors.

        Parameters
        ----------
        v1 : NDArray
            The first vector.
        v2 : NDArray
            The second vector.
        method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.

        Returns
        -------
        corr : float
            The correlation coefficient.
        """
        if method == "spearman":
            corr, _ = spearmanr(v1, v2)
        elif method == "pearson":
            corr = np.corrcoef(v1, v2)[0, 1]
        elif method == "kendall":
            corr, _ = kendalltau(v1, v2)
        else:
            raise ValueError('method must be one of "spearman", "pearson", "kendall"')
        return corr

    def _bootstrap(self, v1: NDArray, v2: NDArray, method: str) -> float:
        """
        Perform one bootstrap iteration to compute correlation.

        Parameters
        ----------
        v1 : NDArray
            The first vector.
        v2 : NDArray
            The second vector.
        method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.

        Returns
        -------
        corr : float
            The correlation coefficient for the bootstrap sample.
        """
        n = len(v1)
        indices = np.random.choice(n, n, replace=True)
        v1_boot = v1[indices]
        v2_boot = v2[indices]
        corr = self._corr(v1_boot, v2_boot, method)
        return corr

    def _permutation(self, v1: NDArray, v2: NDArray, method: str) -> float:
        """
        Perform one permutation iteration to compute correlation.

        Parameters
        ----------
        v1 : NDArray
            The first vector.
        v2 : NDArray
            The second vector.
        method : str
            The correlation method. One of 'spearman', 'pearson', 'kendall'.

        Returns
        -------
        corr : float
            The correlation coefficient for the permuted data.
        """
        v2_perm = np.random.permutation(v2)
        corr = self._corr(v1, v2_perm, method)
        return corr

    def _rdm2vec(self, rdm: NDArray) -> NDArray:
        """
        Convert a square RDM matrix to a vector of its lower triangle elements.

        Parameters
        ----------
        rdm : NDArray
            The RDM matrix.

        Returns
        -------
        vec : NDArray
            The vectorized lower triangle of the RDM.
        """
        lower_triangle = rdm[np.tril_indices(rdm.shape[0], k=-1)]
        return lower_triangle

    def _check_input(
        self,
        rdm1: NDArray,
        rdm2: NDArray,
        input_type: str,
    ) -> None:
        """
        Validate input RDMs.

        Parameters
        ----------
        rdm1 : NDArray
            The first RDM or a set of RDMs.
        rdm2 : NDArray
            The second RDM or a set of RDMs.
        input_type : str
            Type of input RDMs. One of 'n2n', 'one2n', 'one2one'.

        Raises
        ------
        TypeError
            If inputs are not numpy arrays.
        ValueError
            If `input_type` is invalid or RDM dimensions do not match the expected dimensions for the given `input_type`.
        """
        if not isinstance(rdm1, np.ndarray) or not isinstance(rdm2, np.ndarray):
            raise TypeError(
                f"Input RDMs should be numpy arrays, got {type(rdm1)} and {type(rdm2)}"
            )
        if input_type == "n2n":
            if rdm1.ndim != 3 or rdm2.ndim != 3:
                raise ValueError('Both RDMs should be 3D arrays for input_type "n2n"')
        elif input_type == "one2n":
            if rdm1.ndim != 2 or rdm2.ndim != 3:
                raise ValueError(
                    'rdm1 should be a 2D array and rdm2 should be a 3D array for input_type "one2n"'
                )
        elif input_type == "one2one":
            if rdm1.ndim != 2 or rdm2.ndim != 2:
                raise ValueError(
                    'Both RDMs should be 2D arrays for input_type "one2one"'
                )
        else:
            raise ValueError('input_type must be one of "n2n", "one2n", "one2one"')
