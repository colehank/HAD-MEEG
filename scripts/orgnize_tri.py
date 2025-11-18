# %%
from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm_joblib import tqdm_joblib


class MarkerHandler:
    def __init__(self, fp: str | None = None, preload: bool = True):
        self.fp = fp
        self.preload = preload
        self.dtype = None
        if fp is not None:
            self.dtype = "meg" if "meg" in fp.split("/")[-1] else "eeg"
            if self.preload:
                self.load()

    @classmethod
    def from_events(cls, events: np.ndarray, sfreq: float):
        """
        直接从 events array 构造一个 handler：
        events: shape (n_events, 3)
        sfreq: 采样率
        """
        obj = cls(fp=None, preload=False)
        obj.sfreq = sfreq
        obj.events = events
        obj.triggers = events[:, 2]
        unique_vals, counts = np.unique(obj.triggers, return_counts=True)
        obj.trigger_info = dict(zip(unique_vals, counts))
        return obj

    def load(self):
        self.raw = mne.io.read_raw(self.fp)
        self.sfreq = self.raw.info["sfreq"]
        self.events, _ = mne.events_from_annotations(self.raw)
        self.triggers = self.events[:, 2]
        unique_vals, counts = np.unique(self.triggers, return_counts=True)
        self.trigger_info = dict(zip(unique_vals, counts))
        return self.raw

    def _compute_code_stats(self) -> dict[int, dict]:
        stats = {}
        for code, count in self.trigger_info.items():
            idx = np.where(self.triggers == code)[0]
            samples = self.events[idx, 0]
            if len(samples) > 1:
                diffs = np.diff(samples)
                median_diff = np.median(diffs)
                mad = np.median(np.abs(diffs - median_diff))
                variability = mad / median_diff if median_diff > 0 else np.inf
            else:
                median_diff = np.nan
                variability = np.inf
            stats[code] = dict(
                count=int(count),
                idx=idx,
                samples=samples,
                median_diff=float(median_diff),
                variability=float(variability),
            )
        return stats

    def _infer_start_end(
        self,
        stats: dict[int, dict] | None = None,
    ) -> dict[str, int | None]:
        if stats is None:
            stats = self._compute_code_stats()

        counts = {c: s["count"] for c, s in stats.items()}
        max_count = max(counts.values())

        rare_codes = [
            c for c, cnt in counts.items() if cnt <= 3 or cnt <= 0.05 * max_count
        ]

        first_sample = self.events[0, 0]
        last_sample = self.events[-1, 0]

        first_codes = set(self.events[self.events[:, 0] == first_sample, 2])
        last_codes = set(self.events[self.events[:, 0] == last_sample, 2])

        start_candidates = list(first_codes.intersection(rare_codes))
        end_candidates = list(last_codes.intersection(rare_codes))

        start_code = start_candidates[0] if start_candidates else None

        if len(end_candidates) > 1 and start_code in end_candidates:
            end_candidates.remove(start_code)
        end_code = end_candidates[0] if end_candidates else None

        return {
            "begin": start_code,
            "end": end_code,
        }

    def _infer_frame(
        self,
        stats: dict[int, dict] | None = None,
        exclude: set | None = None,
    ) -> int | None:
        if stats is None:
            stats = self._compute_code_stats()
        if exclude is None:
            exclude = set()

        sfreq = getattr(self, "sfreq", None)
        candidates = {
            c: s for c, s in stats.items() if c not in exclude and s["count"] >= 5
        }
        if not candidates:
            return None

        best_code = None
        best_score = -np.inf

        for c, s in candidates.items():
            median_diff = s["median_diff"]
            variability = s["variability"]

            if np.isnan(median_diff) or median_diff <= 0:
                continue

            # 帧间隔必须比较短，比如 < 采样率 / 5 （即 >5Hz）
            if sfreq is not None and median_diff > sfreq / 5.0:
                continue

            score = s["count"] / (1.0 + variability)
            if score > best_score:
                best_score = score
                best_code = c

        if best_code is None:
            best_code = max(
                candidates.keys(),
                key=lambda c: candidates[c]["count"],
            )

        return best_code

    def _infer_resp(
        self,
        stats: dict[int, dict] | None = None,
        fram_code: int | None = None,
        start_end: dict[str, int | None] | None = None,
    ) -> int | None:
        if stats is None:
            stats = self._compute_code_stats()
        if start_end is None:
            start_end = self._infer_start_end(stats)

        exclude = {
            start_end.get("begin"),
            start_end.get("end"),
            fram_code,
        }
        exclude = {c for c in exclude if c is not None}

        candidates = [
            c for c, s in stats.items() if c not in exclude and s["count"] > 1
        ]
        if not candidates:
            return None
        if len(candidates) == 1 or fram_code is None:
            return candidates[0]

        sfreq = getattr(self, "sfreq", None)
        frame_samples = stats[fram_code]["samples"]

        resp_scores = {}
        for c in candidates:
            samples = stats[c]["samples"]
            idx_prev = (
                np.searchsorted(
                    frame_samples,
                    samples,
                    side="right",
                )
                - 1
            )
            valid = idx_prev >= 0
            if not np.any(valid):
                continue

            rt = samples[valid] - frame_samples[idx_prev[valid]]
            rt = rt[rt > 0]
            if rt.size == 0:
                continue

            median_rt = float(np.median(rt))
            mad_rt = float(np.median(np.abs(rt - median_rt)))
            variability = mad_rt / median_rt if median_rt > 0 else np.inf

            in_range_bonus = 1.0
            if sfreq is not None:
                min_rt = 0.1 * sfreq
                max_rt = 1.5 * sfreq
                if not (min_rt <= median_rt <= max_rt):
                    in_range_bonus = 0.3

            score = in_range_bonus * rt.size / (1.0 + variability)
            resp_scores[c] = score

        if resp_scores:
            return max(resp_scores, key=resp_scores.get)
        return candidates[0]

    # ============== 对外：基础 event_id（4 键） ==============
    def infer_event_id(self) -> dict[str, int | None]:
        if not hasattr(self, "events"):
            if self.fp is None:
                raise RuntimeError(
                    "No events found and no raw file path (fp) was provided.",
                )
            self.load()

        stats = self._compute_code_stats()
        start_end = self._infer_start_end(stats)
        fram_code = self._infer_frame(stats, exclude=set(start_end.values()))
        resp_code = self._infer_resp(stats, fram_code, start_end)

        event_id = {
            "begin": start_end.get("begin"),
            "end": start_end.get("end"),
            "fram": fram_code,
            "resp": resp_code,
        }
        return event_id

    def _events_video_start_off(
        self,
        n_videos: int = 90,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        1. 利用 frame marker 自动分出每段视频
        2. 要求恰好 n_videos 段，否则 raise ValueError
        3. 在复制的 events 里：
           - 每段的第一帧 trigger → video on
           - 每段的最后一帧 trigger → video off

        额外约束：
        - 每段 (video off - video on) / sfreq ≈ 2 秒，否则报错

        返回:
            new_events: 修改后的 events
            new_event_id: dict，包含 6 个键：
                'begin', 'end', 'fram', 'resp', 'video on', 'video off'
        """
        if not hasattr(self, "events"):
            if self.fp is None:
                raise RuntimeError(
                    "No events found and no raw file path (fp) was provided.",
                )
            self.load()

        base_event_id = self.infer_event_id()
        fram_code = base_event_id["fram"]
        if fram_code is None:
            raise ValueError(
                "Cannot infer 'fram' code; video segmentation is impossible.",
            )

        stats = self._compute_code_stats()
        frame_stats = stats[fram_code]
        frame_samples = frame_stats["samples"]
        frame_event_idx = frame_stats["idx"]

        if frame_samples.size < 2:
            raise ValueError("Not enough frame markers to segment videos.")

        diffs = np.diff(frame_samples)
        median_diff = np.median(diffs)
        if median_diff <= 0:
            raise ValueError("Cannot infer frame interval from data.")

        # 尝试若干 gap 因子，找到能分成 n_videos 段的那个
        factors = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
        n_by_factor = {}
        chosen_factor = None
        for f in factors:
            gap_thr = median_diff * f
            breaks = np.where(diffs > gap_thr)[0]
            n_segments = breaks.size + 1
            n_by_factor[f] = int(n_segments)
            if n_segments == n_videos:
                chosen_factor = f
                break

        if chosen_factor is None:
            raise ValueError(
                f"Expected {n_videos} videos, but could not segment frame markers into {n_videos} groups. "
                f"Segments by gap factors {factors}: {n_by_factor}",
            )

        gap_thr = median_diff * chosen_factor
        breaks = np.where(diffs > gap_thr)[0]
        start_idx = np.r_[0, breaks + 1]
        end_idx = np.r_[breaks, len(frame_samples) - 1]

        assert len(start_idx) == len(end_idx)
        n_segments = len(start_idx)
        if n_segments != n_videos:
            raise ValueError(
                f"Expected {n_videos} videos, but got {n_segments} after applying threshold factor {chosen_factor}.",
            )

        sfreq = getattr(self, "sfreq", None)
        if sfreq is None or sfreq <= 0:
            raise ValueError(
                "Sampling frequency (sfreq) is not available or invalid.",
            )

        # 每段的首帧 / 末帧 sample
        on_samples = frame_samples[start_idx]
        off_samples = frame_samples[end_idx]
        dur_samples = off_samples - on_samples
        dur_sec = dur_samples / sfreq

        expected_dur = 2.0
        tol = 0.25  # 容忍度（秒），可按需要调，比如 0.2 / 0.3
        lower = expected_dur - tol
        upper = expected_dur + tol

        bad_mask = (dur_sec < lower) | (dur_sec > upper)
        if np.any(bad_mask):
            bad_idx = np.where(bad_mask)[0]
            msg = (
                f"Video duration check failed for {len(bad_idx)} segments. "
                f"Expected ~{expected_dur:.3f}s (±{tol:.3f}s) from video on to video off, "
                f"but got durations in seconds: "
                f"min={dur_sec.min():.3f}, max={dur_sec.max():.3f}. "
                f"Bad segment indices (0-based): {bad_idx.tolist()}"
            )
            raise ValueError(msg)

        new_events = self.events.copy()

        existing_codes = sorted(self.trigger_info.keys())
        max_code = int(existing_codes[-1]) if existing_codes else 0
        video_on_code = max_code + 1
        video_off_code = max_code + 2

        for s_i, e_i in zip(start_idx, end_idx):
            first_event_row = frame_event_idx[s_i]
            last_event_row = frame_event_idx[e_i]
            new_events[first_event_row, 2] = video_on_code
            new_events[last_event_row, 2] = video_off_code

        new_event_id = {
            "begin": base_event_id["begin"],
            "end": base_event_id["end"],
            "fram": base_event_id["fram"],
            "resp": base_event_id["resp"],
            "video on": video_on_code,
            "video off": video_off_code,
        }

        return new_events, new_event_id

    def _simplify_events_for_videos(
        self,
        n_videos: int = 90,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        1) 基于 _events_video_start_off 得到带 video on/off 的 events
        2) events 简化：只保留 video on, video off, begin, end 以及需要的 resp
        3) begin / end 各自只保留一个（按 5 秒规则）：
           - begin: 取第一个视频前 5 秒内出现的第一个 start；
                    若没有，则取全局最早 begin
           - end:   取最后一个视频后 5 秒内出现的第一个 end；
                    若没有，则取全局最晚 end
        4) 对每个 video：
           - 取 video off 之后 1s 内的第一个 resp marker（如果有）
           - 若该窗口内没有 resp，则这个 video 不产生 resp 事件
        5) 自查：video on / video off 数量各 n_videos，
                 且选中的 begin / end 事件确实保留下来
        """
        new_events, full_event_id = self._events_video_start_off(
            n_videos=n_videos,
        )

        start_code = full_event_id["begin"]
        end_code = full_event_id["end"]
        video_on_code = full_event_id["video on"]
        video_off_code = full_event_id["video off"]
        resp_code = full_event_id["resp"]

        if start_code is None:
            raise ValueError(
                "No 'begin' code inferred; cannot enforce single begin.",
            )
        if end_code is None:
            raise ValueError(
                "No 'end' code inferred; cannot enforce single end.",
            )

        triggers = new_events[:, 2]
        samples = new_events[:, 0]

        # 1) 先检查 video on/off 数量
        n_video_on = int(np.sum(triggers == video_on_code))
        n_video_off = int(np.sum(triggers == video_off_code))
        if n_video_on != n_videos or n_video_off != n_videos:
            raise ValueError(
                f"Expected {n_videos} 'video on' and {n_videos} 'video off' events, "
                f"but got {n_video_on} and {n_video_off}.",
            )

        # 2) 视频时间范围
        video_on_samples = samples[triggers == video_on_code]
        video_off_samples = samples[triggers == video_off_code]
        first_video_sample = int(np.min(video_on_samples))
        last_video_sample = int(np.max(video_off_samples))

        sfreq = self.sfreq
        window = int(round(5 * sfreq))  # 5 秒

        # 3) 所有 begin / end 的索引（按 code）
        start_idx_all = np.where(triggers == start_code)[0]
        end_idx_all = np.where(triggers == end_code)[0]

        if start_idx_all.size == 0:
            raise ValueError("No 'begin' events found in events array.")
        if end_idx_all.size == 0:
            raise ValueError("No 'end' events found in events array.")

        # 3.1 选 start：第一个视频前 5s 内的第一个；否则全局最早
        start_window_start = first_video_sample - window
        start_window_end = first_video_sample
        start_candidates = start_idx_all[
            (samples[start_idx_all] >= start_window_start)
            & (samples[start_idx_all] <= start_window_end)
        ]
        if start_candidates.size > 0:
            keep_start_idx = int(
                start_candidates[np.argmin(samples[start_candidates])],
            )
        else:
            keep_start_idx = int(
                start_idx_all[np.argmin(samples[start_idx_all])],
            )

        # 3.2 选 end：最后一个视频后 5s 内的第一个；否则全局最晚
        end_window_start = last_video_sample
        end_window_end = last_video_sample + window
        end_candidates = end_idx_all[
            (samples[end_idx_all] >= end_window_start)
            & (samples[end_idx_all] <= end_window_end)
        ]
        if end_candidates.size > 0:
            keep_end_idx = int(
                end_candidates[np.argmin(samples[end_candidates])],
            )
        else:
            keep_end_idx = int(end_idx_all[np.argmax(samples[end_idx_all])])

        # 4) 构造基础 mask：
        #    先只保留所有 video on/off，再单独把选中的 begin / end 打开
        mask = np.zeros_like(triggers, dtype=bool)
        mask[triggers == video_on_code] = True
        mask[triggers == video_off_code] = True
        mask[keep_start_idx] = True
        mask[keep_end_idx] = True

        # 4.5) 为每个 video 选取一个 resp：video off 之后 1s 内的第一个
        if resp_code is not None and np.any(triggers == resp_code):
            resp_idx_all = np.where(triggers == resp_code)[0]
            resp_samples = samples[resp_idx_all]

            one_sec = int(round(1.0 * sfreq))

            # 全部 video on/off 的索引（按时间顺序）
            video_on_idx = np.where(triggers == video_on_code)[0]
            video_off_idx = np.where(triggers == video_off_code)[0]

            video_on_idx = np.array(video_on_idx, dtype=int)
            video_off_idx = np.array(video_off_idx, dtype=int)

            for i, (on_i, off_i) in enumerate(zip(video_on_idx, video_off_idx)):
                off_sample = samples[off_i]
                upper = off_sample + one_sec

                # 可选：如果下一个 video 很近，把窗口截断到下一个 video on 之前
                if i + 1 < len(video_on_idx):
                    next_on_sample = samples[video_on_idx[i + 1]]
                    if next_on_sample < upper:
                        upper = next_on_sample

                # 找到 (off_sample, upper] 内的所有 resp
                in_win = (resp_samples > off_sample) & (resp_samples <= upper)
                if not np.any(in_win):
                    # 该 video 在 1s 窗口内没有 resp，不新增 resp 事件
                    continue

                cand_idx = resp_idx_all[in_win]
                # 取时间最早的一个
                chosen = cand_idx[np.argmin(samples[cand_idx])]
                mask[chosen] = True

        simplified_events = new_events[mask]

        # 5) 最终检查：video on/off 数量是否仍然正确
        simp_triggers = simplified_events[:, 2]
        n_vo = int(np.sum(simp_triggers == video_on_code))
        n_vf = int(np.sum(simp_triggers == video_off_code))
        if n_vo != n_videos or n_vf != n_videos:
            raise ValueError(
                "Final events count check failed for video on/off: "
                f"video on={n_vo}, video off={n_vf}, expected {n_videos}.",
            )

        # 6) 检查选中的 begin / end 事件确实保留下来
        start_sample = samples[keep_start_idx]
        end_sample = samples[keep_end_idx]

        has_start = np.any(
            (simplified_events[:, 0] == start_sample)
            & (simplified_events[:, 2] == start_code),
        )
        has_end = np.any(
            (simplified_events[:, 0] == end_sample)
            & (simplified_events[:, 2] == end_code),
        )

        if not has_start or not has_end:
            n_s = int(np.sum(simp_triggers == start_code))
            n_e = int(np.sum(simp_triggers == end_code))
            raise ValueError(
                "Final events count check failed: "
                f"start_present={has_start}, end_present={has_end}, "
                f"start_code_count={n_s}, end_code_count={n_e}, "
                f"video on={n_vo}, video off={n_vf}.",
            )

        simplified_event_id = {
            "begin": start_code,
            "end": end_code,
            "video on": video_on_code,
            "video off": video_off_code,
        }

        # 如果确实保留了一些 resp 事件，则在 event_id 中也暴露出去
        if resp_code is not None and np.any(simp_triggers == resp_code):
            simplified_event_id["resp"] = resp_code

        return simplified_events, simplified_event_id

    def infer(
        self,
        order=None,
        n_videos: int = 90,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        总入口：
        1）基于 events 自动推断 begin/end/fram/resp
        2）根据 fram 划分 n_videos 段视频，打上 video on/off
        3）简化 events：只保留 begin / end / video on / video off / resp，
           且按 5 秒规则各自只保留一个 begin/end（语义层面）
        4）如果提供 order dict（例如
           {'begin': 1, 'video on': 2, 'video off': 3, 'resp': 4, 'end': 5}），
           则按“语义”重编码：
             - video on/off/resp 批量改
             - begin/end 即使原始 code 相同，也拆成两个不同的 code
        返回：
            events: 重编码后的简化版 events
            event_id: {'begin', 'end', 'video on', 'video off', 'resp'} → code
                     （若没有 resp 则不含该键）
        """
        simplified_events, simplified_event_id = self._simplify_events_for_videos(
            n_videos=n_videos,
        )

        # 默认编码顺序（不强制包含 resp；如果希望重编码 resp，可在 order 里加 'resp'）
        if order is None or len(order) == 0:
            order = {
                "begin": 1,
                "video on": 2,
                "video off": 3,
                "resp": 4,
                "end": 5,
            }

        start_code = simplified_event_id["begin"]
        end_code = simplified_event_id["end"]
        video_on_code = simplified_event_id["video on"]
        video_off_code = simplified_event_id["video off"]
        resp_code = simplified_event_id.get("resp", None)

        # 基于原始 trigger code 先把各类事件的索引存下来
        triggers0 = simplified_events[:, 2]
        idx_video_on = np.where(triggers0 == video_on_code)[0]
        idx_video_off = np.where(triggers0 == video_off_code)[0]
        idx_resp = (
            np.where(triggers0 == resp_code)[0] if resp_code is not None else None
        )

        # begin / end 的索引
        if start_code == end_code:
            idx_be = np.where(triggers0 == start_code)[0]
            if idx_be.size == 0:
                raise RuntimeError(
                    "Internal error: no events with begin/end code after simplification.",
                )
            idx_begin = int(idx_be[0])
            idx_end = int(idx_be[-1])
        else:
            idx_begin_arr = np.where(triggers0 == start_code)[0]
            idx_end_arr = np.where(triggers0 == end_code)[0]
            idx_begin = (
                int(
                    idx_begin_arr[0],
                )
                if idx_begin_arr.size > 0
                else None
            )
            idx_end = int(idx_end_arr[0]) if idx_end_arr.size > 0 else None

        events_recoded = simplified_events.copy()

        # 1) 批量改 video on/off
        if "video on" in order and idx_video_on.size > 0:
            events_recoded[idx_video_on, 2] = int(order["video on"])
        if "video off" in order and idx_video_off.size > 0:
            events_recoded[idx_video_off, 2] = int(order["video off"])

        # 2) 处理 begin / end
        if idx_begin is not None and "begin" in order:
            events_recoded[idx_begin, 2] = int(order["begin"])
        if idx_end is not None and "end" in order:
            events_recoded[idx_end, 2] = int(order["end"])

        # 3) 处理 resp（如果有，并且 order 中给了）
        resp_code_final = resp_code
        if idx_resp is not None and idx_resp.size > 0 and "resp" in order:
            events_recoded[idx_resp, 2] = int(order["resp"])
            resp_code_final = int(order["resp"])

        # 4) 构造最终的 event_id（和 order 对齐）
        final_event_id = {
            "begin": int(order.get("begin", start_code)),
            "end": int(order.get("end", end_code)),
            "video on": int(order.get("video on", video_on_code)),
            "video off": int(order.get("video off", video_off_code)),
        }
        if resp_code_final is not None:
            final_event_id["resp"] = resp_code_final

        return events_recoded, final_event_id

    def modify(
        self,
        raw: mne.io.BaseRaw,
        events: np.ndarray,
        event_id: dict[str, int],
        keep_bad: bool = True,
    ):
        """
        用给定的 events 和 event_id 更新 raw.annotations。

        参数
        ----
        raw : mne.io.BaseRaw
            原始 Raw 对象，会在函数内被原地修改。
        events : ndarray, shape (n_events, 3)
            目标（修剪后）的 events 数组。
        event_id : dict
            事件名 -> 事件 code 的字典（与你提供的 events 对应）。
        keep_bad : bool, default True
            是否保留原 annotations 中描述以 'BAD' 开头的标记
            （通常是坏段/坏通道等人工标注）。

        返回
        ----
        raw : mne.io.BaseRaw
            更新了 annotations 的 Raw 对象（与输入是同一个对象）。
        """
        sfreq = raw.info["sfreq"]

        # 1. 构造 ID -> 描述 的映射（annotations_from_events 需要这个方向）
        id_to_desc = {code: name for name, code in event_id.items()}

        # 2. 用修剪后的 events 生成新的事件类 annotations（默认 duration=0）
        ann_events = mne.annotations_from_events(
            events=events,
            sfreq=sfreq,
            event_desc=id_to_desc,
            orig_time=raw.info.get("meas_date"),
        )

        # 3. 是否保留原来的 BAD 段等标记
        if keep_bad and raw.annotations is not None and len(raw.annotations) > 0:
            bad_mask = np.array(
                [desc.startswith("BAD") for desc in raw.annotations.description],
            )
            bad_anns = raw.annotations[bad_mask]
            # 合并：原 BAD 段 + 新事件标记
            ann_new = bad_anns + ann_events
        else:
            ann_new = ann_events

        # 4. 写回 raw
        raw.set_annotations(ann_new)

        return raw

    def run(self, fp: str, n_videos: int = 90, order: dict[str, int] | None = None):
        mh = MarkerHandler(fp=fp, preload=True)
        events, event_id = mh.infer(n_videos=n_videos, order=order)
        return self.modify(mh.raw, events, event_id)


def plot_events(raw, title=""):
    events, event_id = mne.events_from_annotations(raw)
    fig = mne.viz.plot_events(
        events,
        sfreq=raw.info["sfreq"],
        show=False,
        event_id=event_id,
    )
    fig.suptitle(title)
    plt.close(fig)
    return fig


def make_events(
    fp: str,
    n_videos: int = 90,
    order: dict[str, int] | None = None,
    onset_short: float = 3.0,
    onset_long: float = 6.0,
    group_size: int = 5,
    video_len: float = 2.0,
    resp_window: float = 1.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    基于实验设计“从 begin/start 开始，5s 后第一个 video on，
    之后按 3s 间隔、每 5 个后 6s 间隔、视频长度 2s”的规则，
    直接构造 events 和 event_id。

    约束（很重要）：
    - 本函数总是按设计生成 n_videos 个 video on 和 n_videos 个 video off
      （例如 n_videos=90 就是 90 对），不会因为录制时间或 trigger 数不够而减少。
    - begin/start 来自真实 trigger：
        * 优先使用 infer_event_id() 返回的 'begin'
        * 若没有 'begin' 键，则使用 'start'
        * 取该 code 最早出现的 sample 作为实验起点
    - video on/off 完全用设计生成（可以超出现有 trigger 的时间范围）
    - resp 只使用已有的 resp trigger：
        * 对每个 trial，取 video off 之后 resp_window 秒内的第一个 resp
        * 若没有，则该 trial 没有 resp 事件（不会伪造）

    返回
    ----
    events : ndarray, shape (n_events, 3)
        生成的 events 数组。
    event_id : dict
        事件名到 code 的映射，至少包含
        {'begin', 'video on', 'video off'}，若存在也包含 'resp' 和 'end'。
    """
    mh = MarkerHandler(fp=fp, preload=True)
    base_event_id = mh.infer_event_id()

    if "begin" in base_event_id and base_event_id["begin"] is not None:
        begin_code = base_event_id["begin"]
    else:
        begin_code = base_event_id.get("start", None)

    end_code = base_event_id.get("end", None)
    resp_code = base_event_id.get("resp", None)

    if begin_code is None:
        raise ValueError(
            "make_events: cannot find 'begin/start' code from data.",
        )

    sfreq = float(mh.sfreq)
    events_raw = mh.events

    begin_samples = events_raw[events_raw[:, 2] == begin_code, 0]
    if begin_samples.size == 0:
        raise ValueError(
            "make_events: no 'begin/start' events found in raw events.",
        )

    begin_sample = int(begin_samples.min())

    if end_code is not None and np.any(events_raw[:, 2] == end_code):
        end_samples = events_raw[events_raw[:, 2] == end_code, 0]
    else:
        end_samples = np.array([], dtype=int)

    if resp_code is not None and np.any(events_raw[:, 2] == resp_code):
        resp_samples_all = events_raw[events_raw[:, 2] == resp_code, 0]
        resp_samples_all = np.sort(resp_samples_all)
    else:
        resp_samples_all = np.array([], dtype=int)

    if order is None or len(order) == 0:
        order = {
            "begin": 1,
            "video on": 2,
            "video off": 3,
            "end": 4,
            "resp": 5,
        }

    begin_out = int(order.get("begin", 1))
    video_on_out = int(order.get("video on", 2))
    video_off_out = int(order.get("video off", 3))
    end_out = int(order.get("end", 4))
    resp_out = int(order.get("resp", 5))

    events_list = []

    # 1) begin 事件（原始 begin/start 第一次出现）
    events_list.append([begin_sample, 0, begin_out])

    on_samples = np.zeros(n_videos, dtype=int)
    on_samples[0] = begin_sample + int(round(5.0 * sfreq))  # begin 后 5 秒

    for k in range(1, n_videos):
        # 间隔：除了每 group_size 个视频之后用 long 间隔，其余用 short 间隔
        if (k % group_size) == 0:
            delta = onset_long
        else:
            delta = onset_short
        on_samples[k] = on_samples[k - 1] + int(round(delta * sfreq))

    off_samples = on_samples + int(round(video_len * sfreq))

    resp_win_samp = int(round(resp_window * sfreq))

    for on_s, off_s in zip(on_samples, off_samples):
        events_list.append([int(on_s), 0, video_on_out])
        events_list.append([int(off_s), 0, video_off_out])

        if resp_samples_all.size > 0:
            # 找到 first resp in (off_s, off_s + resp_win_samp]
            in_win = (resp_samples_all > off_s) & (
                resp_samples_all <= off_s + resp_win_samp
            )
            if np.any(in_win):
                # 第一个真实 resp trigger
                resp_sample = int(resp_samples_all[in_win][0])
                events_list.append([resp_sample, 0, resp_out])

    if end_samples.size > 0:
        # 选在最后一个视频之后出现的第一个 end，如无就取最晚的一个
        last_off = (
            int(
                off_samples.max(),
            )
            if off_samples.size > 0
            else begin_sample
        )
        after_last = end_samples[end_samples >= last_off]
        if after_last.size > 0:
            end_sample = int(after_last[0])
        else:
            end_sample = int(end_samples.max())
        events_list.append([end_sample, 0, end_out])

    events_arr = np.asarray(events_list, dtype=int)
    order_idx = np.argsort(events_arr[:, 0])
    events_arr = events_arr[order_idx]

    trig = events_arr[:, 2]
    n_on = int(np.sum(trig == video_on_out))
    n_off = int(np.sum(trig == video_off_out))

    if n_on != n_videos or n_off != n_videos:
        raise RuntimeError(
            f"make_events internal error: expected {n_videos} 'video on' and "
            f"{n_videos} 'video off', but got {n_on} and {n_off}. "
            "请检查 begin/start 推断是否异常。",
        )

    event_id = {
        "begin": begin_out,
        "video on": video_on_out,
        "video off": video_off_out,
    }
    if resp_samples_all.size > 0:
        event_id["resp"] = resp_out
    if end_samples.size > 0:
        event_id["end"] = end_out

    return events_arr, event_id


def extend_raw_to_fit_events(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    margin_sec: float = 0.0,
    fill_value: float = 0.0,
) -> mne.io.BaseRaw:
    """
    把 raw 的长度强制扩展到足够长，以容纳所有 events 的 sample 索引。

    做法：
    - 读取原 raw 的数据（会自动 preload）
    - 在时间轴末尾拼接一段全 fill_value 的 pad
    - 用同一份 info 和 first_samp 构造一个新的 RawArray

    注意：
    - 返回的是一个新的 Raw 对象（RawArray 类型），
    原始 raw 类型信息会丢失，但采样率/通道/名字等完全一致。
    - 如果 events 里所有 sample 都在原始 raw 范围内，就直接返回原 raw。
    """
    if events is None or events.size == 0:
        return raw

    sfreq = float(raw.info["sfreq"])

    # events 的最大 sample（相对于 raw.first_samp 的索引）
    last_event_sample = int(events[:, 0].max())
    last_event_sample += int(round(margin_sec * sfreq))

    # 当前 raw 的 first_samp / last_samp
    first_samp = int(raw.first_samp)
    n_times = raw.n_times
    last_samp = first_samp + n_times - 1

    # 如果 raw 已经足够长，就不用扩展
    if last_event_sample <= last_samp:
        return raw

    # 需要 pad 的 sample 数
    n_pad = last_event_sample - last_samp
    if n_pad <= 0:
        return raw

    # 取原始数据（自动 preload）
    data = raw.get_data()  # shape: (n_channels, n_times)
    n_ch, n_orig = data.shape

    # 构造 pad 段
    pad_data = np.full((n_ch, n_pad), fill_value, dtype=data.dtype)

    # 拼接原数据 + pad 数据
    new_data = np.concatenate([data, pad_data], axis=1)

    # 复制 info，并保持 first_samp 一致
    info_new = raw.info.copy()

    raw_ext = mne.io.RawArray(
        new_data,
        info_new,
        first_samp=first_samp,
    )

    # 把原来的 annotations 也拷过去（时间是相对 meas_date 的，不受长度影响）
    if raw.annotations is not None and len(raw.annotations) > 0:
        raw_ext.set_annotations(raw.annotations.copy())

    return raw_ext


def run_in_one(
    fp: str,
    n_videos: int = 90,
    order: dict[str, int] | None = {
        "begin": 1,
        "video on": 2,
        "video off": 3,
        "resp": 4,
        "end": 5,
    },
    return_details: bool = True,
):
    mh = MarkerHandler(fp=fp, preload=True)

    try:
        events, event_id = mh.infer(n_videos=n_videos, order=order)
        raw_out = mh.modify(mh.raw, events, event_id)
        info = "success"
    except Exception as e:
        if "Expected 90 videos" in str(e):
            events, event_id = make_events(fp, n_videos=n_videos, order=order)
            try:
                raw = mne.io.read_raw(fp, preload=True)
                raw_ext = extend_raw_to_fit_events(raw, events, margin_sec=0.0)
                new_ann = mne.annotations_from_events(
                    events=events,
                    sfreq=raw.info["sfreq"],
                    event_desc={code: name for name, code in event_id.items()},
                    orig_time=raw_ext.info.get("meas_date"),
                )
                raw_ext.set_annotations(new_ann)
                raw_out = raw_ext
                info = "success after recovered"
            except Exception as e2:
                raise RuntimeError(
                    f"Original error: {str(e)}; Recovery error: {str(e2)}",
                ) from e2
        else:
            raise e

    if return_details:
        return raw_out, info
    else:
        return raw_out


# %%
if __name__ == "__main__":
    with open("for_ev.pkl", "rb") as f:
        for_ev = pickle.load(f)

    dsgn_evid = {
        "begin": 1,
        "video on": 2,
        "video off": 3,
        "resp": 4,
        "end": 5,
    }
    mh = MarkerHandler()
    fig_dir = "/nfs/z1/userhome/zzl-zhangguohao/workingdir/HAD-MEEG_results/figs/marker"
    os.makedirs(fig_dir, exist_ok=True)

    def _process_marker(sub, dtype, run, fp):
        raw, info = run_in_one(
            fp=fp,
            n_videos=90,
            order=dsgn_evid,
            return_details=True,
        )
        fig = plot_events(raw, title=f"sub-{sub}_ses-{dtype}_run-{run}")
        fig.savefig(
            f"{fig_dir}/sub-{sub}_ses-{dtype}_run-{run}_organized_marker.png",
        )
        return None if info == "success" else {"fp": fp, "info": info}

    tasks = [
        (sub, dtype, run, fp)
        for sub, ses in for_ev["meta_info"].items()
        for dtype, runs in ses.items()
        for run, fp in runs.items()
    ]

    with tqdm_joblib(total=len(tasks), desc="Organizing markers"):
        results = Parallel(n_jobs=-1)(delayed(_process_marker)(*task) for task in tasks)

    bad_runs = [res for res in results if res is not None]
    print("Bad runs:", bad_runs)
# %%
