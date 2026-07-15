# %%
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mne.io.constants import FIFF
from mne.transforms import apply_trans, invert_transform
from mne_bids import BIDSPath, read_raw_bids
from src import DataConfig, PlotConfig

cfg_data = DataConfig()
cfg_plot = PlotConfig()
SAVE_DIR = cfg_data.results_root / "basic-head"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = cfg_plot.font_path
FONT_SIZE = cfg_plot.font_size
ELECTRODES = ["nas", "lpa", "rpa"]
N_PARTICIPANTS = cfg_data.source_df.query("datatype == 'meg'")["subject"].nunique()
# CTF fiducial names -> FIFF cardinal identifiers (read from each run's .fif header)
_CARDINAL_IDENT = {
    "nas": FIFF.FIFFV_POINT_NASION,
    "lpa": FIFF.FIFFV_POINT_LPA,
    "rpa": FIFF.FIFFV_POINT_RPA,
}
COLORMAP = "Spectral"
cmap = plt.get_cmap(COLORMAP, 10)
colors = [cmap(i / (10 - 1)) for i in range(10)]
COLOR_MEG = colors[2]
COLOR_EEG = colors[-3]
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()


# %%
class SensorDataLoader:
    """
    Load head-coil fiducial positions (nas/lpa/rpa) for each MEG run.

    Positions are read from every run's ``.fif`` header: the cardinal
    fiducials in ``info['dig']`` (head coordinates) are transformed into the
    device frame via ``info['dev_head_t']``, yielding one coordinate set per
    run -- the same quantity CTF's ``ds.dewar`` provided, but recovered from
    the publicly shared ``.fif`` data (no raw CTF ``.ds`` required). Head
    movement is frame-invariant, so run-to-run motion matches the CTF result
    exactly; only the absolute coordinates live in a different (rigidly
    related) reference frame.

    Parameters
    ----------
    bids_list : List[BIDSPath]
        MEG BIDSPaths, one per run.
    electrodes : List[str]
        Fiducial names to extract (subset of nas/lpa/rpa).

    Attributes
    ----------
    bids_list : List[BIDSPath]
        MEG BIDSPaths to read.
    electrodes : List[str]
        List of fiducial names.
    columns : List[str]
        Column names for the sensor data DataFrame.
    data : pd.DataFrame
        DataFrame to store the sensor positions.
    """

    def __init__(
        self,
        bids_list: List[BIDSPath],
        electrodes: List[str],
    ) -> None:
        self.bids_list = bids_list
        self.electrodes = electrodes
        self.columns = ["subjects", "session", "run"] + [
            f"{elec}_{axis}" for elec in electrodes for axis in ["x", "y", "z"]
        ]
        self.data = pd.DataFrame(columns=self.columns)

    def load_sensor_positions(self) -> pd.DataFrame:
        """
        Load fiducial positions for every MEG run.

        Returns
        -------
        pd.DataFrame
            DataFrame containing fiducial positions for all runs, sorted by
            subject, session, and run.
        """
        all_rows: List[Dict[str, Any]] = []
        for bids in self.bids_list:
            row = self._load_single_run(bids)
            if row:
                all_rows.append(row)
        self.data = (
            pd.DataFrame(all_rows, columns=self.columns)
            .sort_values(["subjects", "session", "run"])
            .reset_index(drop=True)
        )
        return self.data

    def _load_single_run(self, bids: BIDSPath) -> Optional[Dict[str, Any]]:
        """
        Read one run's fiducials from its ``.fif`` header (device frame, cm).

        Parameters
        ----------
        bids : BIDSPath
            BIDSPath of the MEG run.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary of fiducial positions for the run, or None if the file
            is missing or lacks the cardinal fiducials.
        """
        if not bids.fpath.exists():
            print(f"File not found: {bids.fpath}")
            return None

        raw = read_raw_bids(bids, verbose="ERROR")
        cardinals = {
            d["ident"]: d["r"]
            for d in raw.info["dig"] or []
            if d["kind"] == FIFF.FIFFV_POINT_CARDINAL
        }
        head_to_dev = invert_transform(raw.info["dev_head_t"])

        row: Dict[str, Any] = {
            "subjects": int(bids.subject),
            "session": f"ses-{bids.session}",
            "run": int(bids.run),
        }
        for elec in self.electrodes:
            ident = _CARDINAL_IDENT[elec]
            if ident not in cardinals:
                print(f"Missing {elec} fiducial in {bids.fpath}")
                return None
            # head coords (m) -> device/dewar frame -> cm (CTF ds.dewar units)
            x, y, z = apply_trans(head_to_dev, cardinals[ident]) * 100.0
            row[f"{elec}_x"], row[f"{elec}_y"], row[f"{elec}_z"] = x, y, z
        return row


class HeadMovementAnalyzer:
    """
    Class to compute and analyze head movement data.

    Parameters
    ----------
    sensor_data : pd.DataFrame
        DataFrame containing sensor positions.
    electrodes : List[str]
        List of electrode names.
    n_participants : int
        Number of participants.
    results_dir : str
        Directory where results will be saved.
    colors : List[Any]
        List of colors for plotting.
    fontsize : int
        Font size to use in plots.

    Attributes
    ----------
    sensor_data : pd.DataFrame
        DataFrame containing sensor positions.
    head_motion : pd.DataFrame
        DataFrame to store computed head motion data.
    electrodes : List[str]
        List of electrode names.
    n_participants : int
        Number of participants.
    results_dir : str
        Directory where results will be saved.
    colors : List[Any]
        List of colors for plotting.
    fontsize : int
        Font size to use in plots.
    """

    def __init__(
        self,
        sensor_data: pd.DataFrame,
        electrodes: List[str],
        n_participants: int,
        results_dir: str,
        colors: List[Any],
        fontsize: int,
    ) -> None:
        self.sensor_data = sensor_data
        self.head_motion = pd.DataFrame()
        self.electrodes = electrodes
        self.n_participants = n_participants
        self.results_dir = results_dir
        self.colors = colors
        self.fontsize = fontsize

    @staticmethod
    def calculate_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two 3D coordinates.

        Parameters
        ----------
        coord1 : np.ndarray
            First coordinate (3D vector).
        coord2 : np.ndarray
            Second coordinate (3D vector).

        Returns
        -------
        float
            Euclidean distance between the two coordinates.
        """
        return np.linalg.norm(coord1 - coord2)

    def compute_head_movement(self) -> pd.DataFrame:
        """
        Compute head movement within and between sessions for all participants.

        Returns
        -------
        pd.DataFrame
            DataFrame containing head movement data.
        """
        participants = self.sensor_data["subjects"].unique()
        motion_records: List[Dict[str, Any]] = []

        for subjects in participants:
            participant_data = self.sensor_data[
                self.sensor_data["subjects"] == subjects
            ]
            sessions = participant_data["session"].unique()

            # Within-session head movement
            motion_records.extend(
                self._compute_within_session_movement(participant_data, subjects)
            )

            # Between-session head movement
            motion_records.extend(
                self._compute_between_session_movement(
                    participant_data, subjects, sessions
                )
            )

        self.head_motion = pd.DataFrame(motion_records)
        return self.head_motion

    def _compute_within_session_movement(
        self, participant_data: pd.DataFrame, subjects: int
    ) -> List[Dict[str, Any]]:
        """
        Compute within-session head movement for a subjects.

        Parameters
        ----------
        participant_data : pd.DataFrame
            Sensor data for the subjects.
        subjects : int
            subjects number.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing within-session motion records.
        """
        motion_records: List[Dict[str, Any]] = []
        sessions = participant_data["session"].unique()
        for session in sessions:
            session_data = participant_data[participant_data["session"] == session]
            runs = session_data["run"].values
            coords = self._extract_coordinates(session_data)

            for i in range(len(runs) - 1):
                avg_motion = self._compute_average_motion(coords, i, i + 1)
                motion_records.append(
                    {
                        "subjects": subjects,
                        "session": session,
                        "run": runs[i + 1],
                        "motion": avg_motion,
                        "type": "within-session",
                    }
                )
        return motion_records

    def _compute_between_session_movement(
        self, participant_data: pd.DataFrame, subjects: int, sessions: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Compute between-session head movement for a subjects.

        Parameters
        ----------
        participant_data : pd.DataFrame
            Sensor data for the subjects.
        subjects : int
            subjects number.
        sessions : np.ndarray
            Array of session identifiers.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing between-session motion records.
        """
        motion_records: List[Dict[str, Any]] = []
        if len(sessions) > 1:
            for i in range(len(sessions) - 1):
                session1 = sessions[i]
                session2 = sessions[i + 1]
                data1 = participant_data[participant_data["session"] == session1]
                data2 = participant_data[participant_data["session"] == session2]
                coords1 = self._extract_coordinates(data1)
                coords2 = self._extract_coordinates(data2)

                for idx1, run1 in enumerate(data1["run"]):
                    for idx2, run2 in enumerate(data2["run"]):
                        avg_motion = self._compute_average_motion(
                            coords1, idx1, idx2=idx2, coords2=coords2
                        )
                        motion_records.append(
                            {
                                "subjects": subjects,
                                "session1": session1,
                                "session2": session2,
                                "run1": run1,
                                "run2": run2,
                                "motion": avg_motion,
                                "type": "between-session",
                            }
                        )
        return motion_records

    def _extract_coordinates(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract coordinates for electrodes from the data.

        Parameters
        ----------
        data : pd.DataFrame
            Sensor data.

        Returns
        -------
        np.ndarray
            Numpy array of shape (n_runs, n_electrodes, 3) containing coordinates.
        """
        coord_columns = [
            f"{elec}_{axis}" for elec in self.electrodes for axis in ["x", "y", "z"]
        ]
        return data[coord_columns].values.reshape(-1, len(self.electrodes), 3)

    def _compute_average_motion(
        self,
        coords1: np.ndarray,
        idx1: int,
        idx2: Optional[int] = None,
        coords2: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute average motion between two sets of coordinates.

        Parameters
        ----------
        coords1 : np.ndarray
            Coordinates from the first data set.
        idx1 : int
            Index of the first coordinate set.
        idx2 : Optional[int], default=None
            Index of the second coordinate set.
        coords2 : Optional[np.ndarray], default=None
            Coordinates from the second data set.

        Returns
        -------
        float
            Average motion between the two coordinate sets.
        """
        if coords2 is None:
            coords2 = coords1
            idx2 = idx2 if idx2 is not None else idx1 + 1
        distances = [
            self.calculate_distance(coords1[idx1][j], coords2[idx2][j])
            for j in range(len(self.electrodes))
        ]
        return np.mean(distances)

    def plot_head_motion(self, figsize=(12, 3)) -> None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        _plot_head_motion = self.head_motion[
            self.head_motion["motion"] <= 2
        ].reset_index(drop=True)
        print(f"droped cols: \n{self.head_motion[self.head_motion['motion'] > 2]}")
        sns.violinplot(
            data=_plot_head_motion,
            x="subjects",
            y="motion",
            hue="type",
            inner=None,
            ax=ax,
            linewidth=0,
            density_norm="width",
            palette=self.colors,
            legend=False,
            native_scale=True,
            cut=0,
        )
        sns.boxplot(
            data=_plot_head_motion,
            x="subjects",
            y="motion",
            showcaps=True,
            boxprops={
                "facecolor": "dimgray",
                "edgecolor": "dimgray",
                "alpha": 0.6,
                "linewidth": 0.8,
            },
            showfliers=False,  # 这里顺便关掉极端离群点的marker
            whiskerprops={"linewidth": 1.0, "color": "dimgray"},
            medianprops={"color": "lightgoldenrodyellow", "linewidth": 2.2},
            width=0.15,
            ax=ax,
            native_scale=True,
        )

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(0, 31)
        ax.set_xticks(np.arange(1, self.n_participants + 1))
        # ax.set_xticklabels(format(i, '02d') for i in range(1, self.n_participants + 1))
        ax.set_ylim(0, 0.8)
        ax.set_yticks(np.arange(0, 0.9, 0.4))
        ax.tick_params(axis="x", labelsize=self.fontsize - 2)
        ax.tick_params(axis="y", labelsize=self.fontsize - 2)
        ax.set_xlabel("Subject", fontsize=self.fontsize)
        ax.set_ylabel("Head motion (mm)", fontsize=self.fontsize)
        plt.tight_layout()
        return fig

    def print_motion_statistics(self) -> None:
        """
        Calculate and print median head motion statistics.

        Returns
        -------
        None
            The function prints the statistics to the console.
        """
        within_motion = self.head_motion[self.head_motion["type"] == "within-session"]
        between_motion = self.head_motion[self.head_motion["type"] == "between-session"]

        within_sub = within_motion.groupby("subjects")["motion"].mean()
        between_sub = between_motion.groupby("subjects")["motion"].mean()

        print(
            f"Within-session: median {within_sub.median():.3f} mm, mean {within_sub.mean():.3f} mm, std {within_sub.std():.3f} mm"
        )
        print(
            f"Between-session: median {between_sub.median():.3f} mm, mean {between_sub.mean():.3f} mm, std {between_sub.std():.3f} mm"
        )


def plot_head_motion(self, figsize=(12, 3)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # 过滤 motion
    df = self.head_motion[self.head_motion["motion"] <= 2].reset_index(
        drop=True
    )  # for the plot's scale will be better
    print(f"droped cols: \n{self.head_motion[self.head_motion['motion'] > 2]}")

    subjects = sorted(df["subjects"].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    df = df.copy()
    df["subj_idx"] = df["subjects"].map(subj_to_idx)

    sns.barplot(
        data=df,
        x="subj_idx",
        y="motion",
        estimator=np.median,
        errorbar=None,
        width=0.6,
        color=COLOR_MEG,
        alpha=1,
        ax=ax,
        zorder=1,
    )

    rng = np.random.default_rng(0)
    jitter = 0.1

    for i, s in enumerate(subjects):
        df_sub = df[df["subjects"] == s]

        for _, row in df_sub.iterrows():
            x_jitter = rng.uniform(-jitter, jitter)
            x_val = i + x_jitter
            y_val = row["motion"]
            ax.scatter(
                x_val,
                y_val,
                s=10,
                alpha=0.6,
                color="lightgray",
                edgecolor="dimgray",
                zorder=2,
            )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(-0.5, self.n_participants - 0.5)
    ax.set_xticks(np.arange(self.n_participants))
    ax.set_xticklabels(np.arange(1, self.n_participants + 1))
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.arange(0, 0.9, 0.4))
    ax.tick_params(axis="x", labelsize=self.fontsize - 2)
    ax.tick_params(axis="y", labelsize=self.fontsize - 2)
    ax.set_xlabel("Subject", fontsize=self.fontsize)
    ax.set_ylabel("Head motion (mm)", fontsize=self.fontsize)
    plt.tight_layout()
    return fig


# %%
if __name__ == "__main__":
    meg_bids = [bp for bp in cfg_data.source_bids_list if bp.datatype == "meg"]
    data_loader = SensorDataLoader(bids_list=meg_bids, electrodes=ELECTRODES)
    sensor_data = data_loader.load_sensor_positions()

    cmap = plt.get_cmap(COLORMAP, 10)
    colors = [cmap(i / (10 - 1)) for i in range(10)]
    colors = [colors[2], colors[-3]]
    fontsize = 20
    analyzer = HeadMovementAnalyzer(
        sensor_data=sensor_data,
        electrodes=ELECTRODES,
        n_participants=N_PARTICIPANTS,
        results_dir=SAVE_DIR,
        colors=colors,
        fontsize=fontsize,
    )
    analyzer.compute_head_movement()
    analyzer.print_motion_statistics()

    sensor_data.to_csv(SAVE_DIR / "MEG-head_motion.csv", index=False)
    # %%
    fig = plot_head_motion(analyzer)
    fig.savefig(
        SAVE_DIR / "MEG-head_motion.png", dpi=300, bbox_inches="tight", transparent=True
    )
    fig.savefig(
        SAVE_DIR / "MEG-head_motion.svg", dpi=300, bbox_inches="tight", transparent=True
    )

# %%
