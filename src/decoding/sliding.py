# %%
import numpy as np
from mne import BaseEpochs
from mne.decoding import (
    UnsupervisedSpatialFilter,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
)
from ..utils import get_soi_picks
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


# %%
class EpochsDecoder:
    def __init__(
        self,
        epochs: list[BaseEpochs, BaseEpochs],
        pick_type: str = "all",
        random_state: int = 97,
    ):
        self.pick_type = pick_type
        self.info = epochs[0].info
        self.random_state = random_state

        epochs = self._preprocess_epochs(epochs)
        X = np.concatenate((epochs[0]._data, epochs[1]._data), axis=0)
        y = np.concatenate(
            (
                np.ones(len(epochs[0])),
                np.zeros(len(epochs[1])),
            )
        )
        self.X, self.y = shuffle(X, y, random_state=self.random_state)

    def sliding_svc(
        self,
        scoring: str,
        cv: int | str = 10,
        pca: bool = False,
        C: float = 0.5,
        tol: float = 1e-3,
        pca_threshold: float = 0.95,
        n_jobs: int = 1,
    ) -> np.ndarray:
        X = self.X
        y = self.y

        steps = [Scaler(info=self.info)]  # 3D: (epochs, ch, time)

        if pca:
            spatial_pca = UnsupervisedSpatialFilter(
                PCA(n_components=pca_threshold, svd_solver="full"),
                average=False,
            )
            steps.append(spatial_pca)

        steps += [
            Vectorizer(),  # 2D: (epochs, features)
            LinearSVC(
                C=C,
                random_state=self.random_state,
                class_weight="balanced",
                tol=tol,
                max_iter=99999,
            ),
        ]
        clf = make_pipeline(*steps)

        if cv == "loo":
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(
                n_splits=cv, random_state=self.random_state, shuffle=True
            )
        time_decod = SlidingEstimator(
            clf,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=None,
        )

        scores = cross_val_multiscore(
            time_decod,
            X,
            y,
            cv=cv,
            n_jobs=1,
            verbose=None,
        )
        return np.mean(scores, axis=0)

    @staticmethod
    def _pca_on_channel(X: np.ndarray, threshold: float) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("X should be 3D array, (n_epochs, n_channels, n_times)")

        PCA_ = UnsupervisedSpatialFilter(PCA(threshold), average=False)
        return PCA_.fit_transform(X)

    def _preprocess_epochs(
        self, epos: list[BaseEpochs], drop_chans: list[str] = ["M1", "M2"]
    ) -> list[BaseEpochs]:
        epochs = []
        for epo in epos:
            if any(chan in epo.ch_names for chan in drop_chans):
                epo.drop_channels(ch_names=drop_chans)
            epo = epo.pick(get_soi_picks(epo, self.pick_type))
            epochs.append(epo)
        return epochs

    def _repr_html_(self):
        import pandas as pd

        to_show = {
            "nSamples": len(self.X),
            "sensorsType": self.pick_type,
            "nChannels": len(self.info["ch_names"]),
            "randomState": self.random_state,
        }
        to_show = pd.DataFrame(to_show, index=[0]).T
        to_show.columns = ["SVC_Classifier"]
        return to_show.to_html()
