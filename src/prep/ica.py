# %%
from __future__ import annotations

import contextlib
import io
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne_icalabel import label_components

from ..viz import plot_2d_topo
from ._bids_loader import BaseLoader


class ICLabels(TypedDict):
    labels: Sequence[str]  # list[str], length n_components
    y_pred_proba: np.ndarray  # shape (n_components,)
    methods: Sequence[str] | None  # list[str], length n_components


class ICARunner(BaseLoader):
    def __init__(
        self,
        raw: BaseRaw,
        dtype: str | None = None,
        random_state: int = 42,
    ):
        super().__init__(raw, dtype)
        self.random_state = random_state

    def _prep_raw(
        self,
        raw: BaseRaw,
        highpass: float | None = 1,
        lowpass: float | None = None,
        sfreq: float | None = None,
    ) -> BaseRaw:
        """prepare raw data for ICA decomposition"""
        if lowpass is None:
            lowpass = raw.info["lowpass"]
        if sfreq is None:
            sfreq = raw.info["sfreq"]
        if highpass is None:
            highpass = raw.info["highpass"]

        assert highpass < lowpass <= sfreq / 2, (
            f"lowpass {lowpass} must be less than Nyquist {int(sfreq) / 2} "
            f"and higher than highpass {highpass}."
        )

        if sfreq != 250.0:
            logger.warning(
                f"megnet and iclabel models are trained on 250 Hz data. "
                f"{sfreq} Hz data may lead to suboptimal performance.",
            )

        _raw = raw.copy()
        _raw.load_data()
        _raw = self._pick_chs(_raw)
        _raw.resample(sfreq)
        _raw.filter(highpass, lowpass)
        return _raw

    def _feature_extra(
        self,
        raw: BaseRaw,
        n_comp: int | float | None = None,
        method: str = "infomax",
        fit_params: dict | None = None,
    ) -> ICA:
        logger.trace("Extracting ICA components...")
        if n_comp is None:
            if self.dtype == "eeg":
                n_comp = 20
            elif self.dtype == "meg":
                n_comp = 40

        if fit_params is None and method in ("infomax", "picard"):
            fit_params = dict(extended=True)

        ica = ICA(
            n_components=n_comp,
            method=method,
            random_state=self.random_state,
            max_iter="auto",
            fit_params=fit_params if method in ("infomax", "picard") else None,
        )
        ica.fit(raw)
        return ica

    def label_ica_components_auto(
        self,
        raw: BaseRaw,
        ica: ICA,
        method: str | None = None,
        good_labels: list[str] | None = None,
    ) -> tuple[ICA, ICLabels]:
        """label ICA components using mne-icalabel"""
        logger.trace("Labeling ICA components automatically...")
        if method is None:
            method = "iclabel" if self.dtype == "eeg" else "megnet"
        if good_labels is None:
            good_labels = ["brain", "other", "brain/other"]

        # run auto label
        ic_labels = label_components(raw, ica, method=method)
        ic_labels["methods"] = [method] * ica.n_components_

        # write label to ICA
        ic_labels_dict = {
            label: [
                int(idx)
                for idx, val in enumerate(
                    ic_labels["labels"],
                )
                if val == label
            ]
            for label in set(ic_labels["labels"])
        }
        ica.labels_ = ic_labels_dict

        # write exclude to ICA
        exclude = [
            int(idx)
            for idx, val in enumerate(
                ic_labels["labels"],
            )
            if val not in good_labels
        ]
        ica.exclude = exclude
        return ica, ic_labels

    def label_ica_components_manual(
        self,
        ica: ICA,
        auto_labels: ICLabels,
        manual_labels: list[str],
    ) -> tuple[ICA, ICLabels]:
        """label ICA components manually"""
        logger.trace("Labeling ICA components manually...")
        if len(manual_labels) != ica.n_components_ != len(auto_labels["labels"]):
            raise ValueError(
                f"Number of labels ({len(manual_labels)}) does not match "
                f"number of ICA components ({ica.n_components_}).",
            )
        ic_labels_dict = {
            label: [
                int(idx)
                for idx, val in enumerate(
                    manual_labels,
                )
                if val == label
            ]
            for label in set(manual_labels)
        }
        ica.labels_ = ic_labels_dict
        exclude = [
            int(idx)
            for idx, val in enumerate(manual_labels)
            if val not in ["brain", "other", "brain/other"]
        ]

        ica.exclude = exclude

        auto_pred_proba = auto_labels["y_pred_proba"]
        save_pred_proba = auto_pred_proba.copy()
        merged_labels = []
        merged_pred = []
        merged_methods = []
        for manual_label, auto_label, auto_prob in zip(
            manual_labels,
            auto_labels["labels"],
            save_pred_proba,
        ):
            if manual_label == auto_label:
                merged_labels.append(auto_label)
                merged_pred.append(auto_prob)
                merged_methods.append(
                    "MEGNet" if self.dtype == "meg" else "ICLabel",
                )
            else:
                merged_labels.append(manual_label)
                merged_pred.append(1.0)
                merged_methods.append("Manual")

        match_auto_labels = {
            "labels": merged_labels,
            "y_pred_proba": np.array(merged_pred),
            "methods": merged_methods,
        }
        return ica, match_auto_labels

    def plot_comps_with_labels(
        self,
        ica: ICA,
        raw: BaseRaw,
        ic_labels: ICLabels,
        sphere: float | None = None,
    ) -> plt.Figure:
        """plot ICA components with labels"""
        logger.trace("Plotting ICA components with labels...")
        if sphere is None:
            sphere = 0.2 if self.dtype == "meg" else 0.15
            logger.trace(
                f"sphere is not provided, set to {sphere} for {self.dtype} data.",
            )
        labels = ic_labels["labels"]
        y_pred = ic_labels["y_pred_proba"]

        n_comp = ica.n_components_
        n_row = min(n_comp, 4)
        n_col = math.ceil(n_comp / n_row)
        fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row))
        axes = axes.flatten()
        for i in range(len(axes)):
            if i >= n_comp:
                axes[i].axis("off")
                continue
            data = ica.get_components()[:, i]
            plot_2d_topo(
                data,
                raw,
                sphere=sphere,
                sensors=False,
                contours=0,
                axes=axes[i],
                show=False,
                cmap="RdBu_r",
            )
            label = labels[i]
            pred = y_pred[i]
            pred = f"{pred:.2f}"
            color = (
                "red"
                if label
                not in [
                    "brain",
                    "other",
                    "brain/other",
                ]
                else "black"
            )
            axes[i].set_title(
                f"{i}\n{label}({pred})",
                color=color,
                fontsize=20,
            )

        fig.subplots_adjust(hspace=0.5)
        plt.close(fig)
        return fig

    def plot_comp_with_label(
        self,
        ica: ICA,
        raw: BaseRaw,
        ic_idx: int,
        labels: ICLabels,
    ) -> plt.Figure:
        """plot single ICA component with label"""
        fig = plt.figure(figsize=[8, 8])
        axes_paras = (
            ("topomap", [0.08, 0.55, 0.3, 0.4]),
            ("image", [0.5, 0.65, 0.45, 0.25]),
            ("erp", [0.5, 0.55, 0.45, 0.1]),
            ("spectrum", [0.08, 0.15, 0.32, 0.35]),
            ("variance", [0.5, 0.15, 0.45, 0.25]),
            ("source", [0.08, 0, 0.88, 0.05]),
        )
        axes = [fig.add_axes(loc, label=name) for name, loc in axes_paras]
        with contextlib.redirect_stdout(io.StringIO()):  # mute for simplicity
            ica.plot_properties(raw, picks=ic_idx, axes=axes[:5], show=False)

        data = ica.get_sources(raw).get_data(picks=ic_idx)
        axes[5].set_title("Source Time Series (first 15 seconds)")
        times = raw.times
        smp_fifteen_sec = int(15 * raw.info["sfreq"])
        data = data[:, :smp_fifteen_sec]
        times = times[:smp_fifteen_sec]
        axes[5].plot(times, data[0], color="black")
        axes[5].set_xlabel("Time (s)")
        axes[5].set_ylabel("Amplitude")
        label = labels["labels"][ic_idx]
        ypred = labels["y_pred_proba"][ic_idx]
        pred = f"{ypred:.2f}"
        color = (
            "red"
            if label
            not in [
                "brain",
                "other",
                "brain/other",
            ]
            else "black"
        )
        axes[0].set_title(f"{ic_idx}\n{label}({pred})", color=color)
        plt.close(fig)
        return fig

    def _save_deriv(
        self,
        ica: ICA,
        raw: BaseRaw,
        labels: ICLabels,
        fname: str,
        save_keywargs: dict | None = None,
        author: str | None = "n/a",
    ) -> None:
        """save ICA component plots to file"""
        logger.trace(f"Saving ICA derivative files to {fname} ...")
        if save_keywargs is None:
            save_keywargs = dict(
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
        plt.close("all")
        fig_all = self.plot_comps_with_labels(ica, raw, labels)
        fig_all.savefig(f"{fname}.png", **save_keywargs)
        del fig_all

        comps_dir = Path(fname)
        os.makedirs(comps_dir, exist_ok=True)
        for i in range(ica.n_components_):
            fig_single = self.plot_comp_with_label(ica, raw, i, labels)
            fig_single.savefig(f"{comps_dir}/comp-{i}.png", **save_keywargs)
            del fig_single
        plt.close("all")

        status = ["good"] * ica.n_components_
        status_description = ["n/a"] * ica.n_components_
        ic_type = ["n/a"] * ica.n_components_

        if ica.labels_:
            for label, comps in ica.labels_.items():
                this_status = (
                    "good"
                    if label
                    in [
                        "brain",
                        "other",
                        "brain/other",
                    ]
                    else "bad"
                )
                for comp in comps:
                    status[comp] = this_status
                    ic_type[comp] = label

        tsv_data = pd.DataFrame(
            dict(
                component=list(range(ica.n_components_)),
                type=["ica"] * ica.n_components_,
                description=["Independent Component"] * ica.n_components_,
                status=status,
                status_description=status_description,
                annotate_method=labels["methods"],
                annotate_author=[author] * ica.n_components_,
                ic_type=ic_type,
                pred_prob=labels["y_pred_proba"].tolist(),
            ),
        )
        component_json = {
            "annotate_method": "Method used for annotating components (e.g. Manual, ICLabel)",
            "annotate_author": "The name of the person who ran the annotation",
            "ic_type": "The type of annotation must be one of ['brain', "
            "'muscle artifact', 'eye blink', 'heart beat', 'line noise', "
            "'channel noise', 'other']",
            "pred_prob": "The predicted probability of the component belonging to the annotated type,"
            "if annotated manually, this value is 1.0",
        }

        tsv_data.to_csv(
            f"{fname}.tsv",
            sep="\t",
            index=False,
            encoding="utf-8",
        )
        with open(f"{fname}.json", "w", encoding="utf-8") as jf:
            json.dump(component_json, jf, indent=4)
        ica.save(f"{fname}.fif", overwrite=True)

    def load_lc_labels(
        self,
        fname: str,
    ) -> ICLabels:
        data_df = pd.read_csv(fname, sep="\t")
        labels = data_df["ic_type"].tolist()
        pred_prob = data_df["pred_prob"].to_numpy()
        methods = data_df["annotate_method"].tolist()
        return ICLabels(
            labels=labels,
            y_pred_proba=pred_prob,
            methods=methods,
        )

    def regress_artifacts(
        self,
        ica: ICA,
        raw: BaseRaw,
        exclude: list[int] | None = None,
        highpass: float | None = 0.1,
        lowpass: float | None = 100,
        sfreq: float | None = 250,
    ) -> BaseRaw:
        """regress out artifact components from raw data"""
        logger.trace("Regressing out artifact components...")
        if highpass is None:
            highpass = ica.info["highpass"]
        if lowpass is None:
            lowpass = ica.info["lowpass"]
        if sfreq is None:
            sfreq = ica.info["sfreq"]
        _raw = self._prep_raw(
            raw,
            highpass=highpass,
            lowpass=lowpass,
            sfreq=sfreq,
        )

        if exclude is not None:
            ica.exclude = exclude
        return ica.apply(_raw)

    def run(
        self,
        regress: bool = False,
        # raw params for ica input
        highpass: float | None = 1.0,
        lowpass: float | None = 100.0,
        sfreq: float | None = 250,
        # raw params for ica regression
        highpass_regress: float | None = 0.1,
        lowpass_regress: float | None = 100.0,
        sfreq_regress: float | None = 250,
        # ICA params
        ncomp: int | float | None = None,
        # Pipline control
        manual: bool = False,
        manual_labels: list[str] | None = None,
        auto_labels: ICLabels | str | None = None,
        ica: ICA | None = None,
        # derivative saving params
        save_deriv: bool = True,
        fname: str = "ica_output",
    ) -> None | BaseRaw:
        """Full pipeline: prepare raw data, extract ICA components
        A standard pipeline need user to run auto labeling first,
        then check the labeling result and do manual labeling if necessary,
        finally regress out artifact components.

        Parameters
        ----------
        regress : bool, optional
            whether to regress out artifact components, by default False
        highpass : float, optional
            highpass filter for ICA decomposition, by default 1.0
        lowpass : float, optional
            lowpass filter for ICA decomposition, by default 100.0
        sfreq : float, optional
            resampling frequency for ICA decomposition, by default 250
        highpass_regress : float, optional
            highpass filter for ICA regression, by default .1
        lowpass_regress : float, optional
            lowpass filter for ICA regression, by default 100.0
        sfreq_regress : float, optional
            resampling frequency for ICA regression, by default 250
        ncomp : int or float, optional
            number of ICA components, by default 20 for eeg and 40 for meg
        manual : bool, optional
            whether to do manual labeling, by default False
        ic_labels : list of str, optional
            manual labels for ICA components, required if manual is True
        ica : ICA, optional
            precomputed ICA object, required if manual is True
        save_deriv : bool, optional
            whether to save ICA derivative files, by default True
        save_name : str, optional
            base name for saving ICA derivative files, by default 'ica_output'
        """
        logger.info(
            f"Running ICA {'manual' if manual else 'auto'} pipeline...",
        )
        if regress and not manual:
            logger.warning(
                "Regressing artifacts without manual checking."
                "Please make sure the automatic"
                "labeling is reliable before regression.",
            )
        raw = self.raw.copy()
        if any(
            val != val_regress
            for val, val_regress in zip(
                (lowpass, sfreq),
                (lowpass_regress, sfreq_regress),
            )
        ):
            logger.warning(
                "ICA decomposition and regression use different filtering "
                "and resampling settings.",
            )
        in_raw = self._prep_raw(raw, highpass, lowpass, sfreq)
        match manual:
            case False:
                ica = self._feature_extra(in_raw, ncomp)
                ica, ic_labels = self.label_ica_components_auto(
                    in_raw,
                    ica,
                )
            case True:
                if ica is None or manual_labels is None:
                    raise ValueError(
                        "manual labeling requires both ica and manual_labels provided.",
                    )

                if auto_labels is None:
                    try:
                        auto_labels_path = (
                            str(
                                fname,
                            )
                            + f"_desc-ica_{self.dtype}.tsv"
                        )
                        auto_labels = self.load_lc_labels(auto_labels_path)
                    except Exception as e:
                        raise ValueError(
                            "Please provide auto_labels path or ICLabels object for manual labeling.",
                            e,
                        ) from e

                elif isinstance(auto_labels, str):
                    auto_labels_path = auto_labels
                    auto_labels = self.load_lc_labels(auto_labels_path)

                ica, ic_labels = self.label_ica_components_manual(
                    ica,
                    auto_labels=auto_labels,
                    manual_labels=manual_labels,
                )

        if save_deriv:
            if fname is None:
                raise ValueError(
                    "Please provide a filename to save the derivative.",
                )
            fname = f"{fname}_desc-ica_{self.dtype}"
            os.makedirs(Path(fname).parent, exist_ok=True)
            self._save_deriv(
                ica,
                in_raw,
                ic_labels,
                fname,
            )
            logger.trace(f"ICA derivative files saved to {fname}.*")

        if regress:
            reg_raw = self.regress_artifacts(
                ica,
                raw,
                highpass=highpass_regress,
                lowpass=lowpass_regress,
                sfreq=sfreq_regress,
            )
            logger.trace("Artifact components regressed out from raw data.")
            return reg_raw
