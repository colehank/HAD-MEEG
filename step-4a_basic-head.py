# %%
import os
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyctf import dsopen
from matplotlib import font_manager as fm
from src import DataConfig, PlotConfig
from pathlib import Path

cfg_data = DataConfig()
cfg_plot = PlotConfig()
DATA_DIR = Path(
    "/nfs/z1/userhome/zzl-zhangguohao/workingdir/BIN/action/HAD-MEEG-BIDS"
)  # REFERING TO CTF BIDS
SAVE_DIR = cfg_data.results_root / "basic-head"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = cfg_plot.font_path
FONT_SIZE = cfg_plot.font_size
ELECTRODES = ["nas", "lpa", "rpa"]
N_PARTICIPANTS = cfg_data.source_df.query("datatype == 'meg'")["subject"].nunique()
SESSIONS_CONFIG = {
    range(1, 31): {
        "ses-meg": 8,
    }
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
    Class to load and store sensor positions for participants.

    Parameters
    ----------
    root_dir : str
        Root directory containing MEG data files.
    n_participants : int
        Number of participants.
    sessions_config : Dict[int, Dict[str, int]]
        Dictionary mapping subjects ranges to their session configurations.
    electrodes : List[str]
        List of electrode names to consider.

    Attributes
    ----------
    root_dir : str
        Root directory containing MEG data files.
    n_participants : int
        Number of participants.
    sessions_config : Dict[int, Dict[str, int]]
        Session configurations for participants.
    electrodes : List[str]
        List of electrode names.
    columns : List[str]
        Column names for the sensor data DataFrame.
    data : pd.DataFrame
        DataFrame to store the sensor positions.
    """

    def __init__(
        self,
        root_dir: str,
        n_participants: int,
        sessions_config: Dict[int, Dict[str, int]],
        electrodes: List[str],
    ) -> None:
        self.root_dir = root_dir
        self.n_participants = n_participants
        self.sessions_config = sessions_config
        self.electrodes = electrodes
        self.columns = ["subjects", "session", "run"] + [
            f"{elec}_{axis}" for elec in electrodes for axis in ["x", "y", "z"]
        ]
        self.data = pd.DataFrame(columns=self.columns)

    def load_sensor_positions(self) -> pd.DataFrame:
        """
        Load sensor positions for all participants, sessions, and runs.

        Returns
        -------
        pd.DataFrame
            DataFrame containing sensor positions for all runs.
        """
        all_rows: List[Dict[str, Any]] = []
        for subjects in range(1, self.n_participants + 1):
            participant_sessions = self._get_participant_sessions(subjects)
            for session, n_runs in participant_sessions.items():
                for run in range(1, n_runs + 1):
                    row = self._load_single_run(subjects, session, run, task="action")
                    if row:
                        all_rows.append(row)
        self.data = pd.DataFrame(all_rows, columns=self.columns)
        return self.data

    def _get_participant_sessions(self, subjects: int) -> Dict[str, int]:
        """
        Get the session configuration for a subjects.

        Parameters
        ----------
        subjects : int
            subjects number.

        Returns
        -------
        Dict[str, int]
            Session configuration for the subjects.
        """
        for participant_range, sessions in self.sessions_config.items():
            if subjects in participant_range:
                return sessions
        return {}

    def _load_single_run(
        self, subjects: int, session: str, run: int, task: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load sensor positions for a single run.

        Parameters
        ----------
        subjects : int
            subjects number.
        session : str
            Session identifier.
        run : int
            Run number.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing sensor positions for the run,
            or None if the file is not found.
        """
        meg_fn = (
            f"{self.root_dir}/sub-{subjects:02d}/{session}/meg/"
            f"sub-{subjects:02d}_{session}_task-{task}_run-{run:02}_meg.ds"
        )
        if os.path.exists(meg_fn):
            ds = dsopen(meg_fn)
            row: Dict[str, Any] = {
                "subjects": subjects,
                "session": session,
                "run": run,
            }
            for i, elec in enumerate(self.electrodes):
                row.update(
                    {
                        f"{elec}_x": ds.dewar[i][0],
                        f"{elec}_y": ds.dewar[i][1],
                        f"{elec}_z": ds.dewar[i][2],
                    }
                )
            return row
        else:
            print(f"File not found: {meg_fn}")
            return None


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
    data_loader = SensorDataLoader(
        root_dir=DATA_DIR,
        n_participants=N_PARTICIPANTS,
        sessions_config=SESSIONS_CONFIG,
        electrodes=ELECTRODES,
    )
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
