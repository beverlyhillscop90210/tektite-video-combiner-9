import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import folder_paths
import torch
from PIL import Image

try:
    from comfy_api.latest import InputImpl
except Exception:  # pragma: no cover
    InputImpl = None

MAX_INPUTS = 16
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".m2ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class TektiteVideoCombiner9:
    CATEGORY = "Tektite/Video"
    FUNCTION = "combine_videos"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs: Dict[str, Any] = {}
        optional_inputs["audio"] = ("AUDIO",)
        for i in range(1, MAX_INPUTS + 1):
            optional_inputs[f"clip{i}"] = ("*",)

        return {
            "required": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "output file path or output folder",
                    },
                ),
                "output_format": (["mp4", "mov", "mkv"], {"default": "mp4"}),
                "sequence_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "wait_timeout_sec": ("INT", {"default": 300, "min": 1, "max": 300, "step": 1}),
                "poll_interval_sec": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "stable_polls": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "reencode_fallback": ("BOOLEAN", {"default": True}),
                "video_codec": ("STRING", {"default": "libx264", "multiline": False}),
                "preset": (
                    [
                        "ultrafast",
                        "superfast",
                        "veryfast",
                        "faster",
                        "fast",
                        "medium",
                        "slow",
                        "slower",
                        "veryslow",
                    ],
                    {"default": "medium"},
                ),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
            },
            "optional": optional_inputs,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def combine_videos(
        self,
        output_path: str,
        output_format: str,
        sequence_fps: float,
        wait_timeout_sec: int,
        poll_interval_sec: float,
        stable_polls: int,
        overwrite: bool,
        reencode_fallback: bool,
        video_codec: str,
        preset: str,
        crf: int,
        audio=None,
        **kwargs,
    ):
        if InputImpl is None:
            raise RuntimeError("This node needs a recent ComfyUI build with native VIDEO support.")
        # Runtime cache for in-memory VIDEO inputs (no filesystem path).
        self._runtime_video_cache: Dict[int, str] = {}
        self._runtime_image_cache: Dict[int, List[str]] = {}
        self._runtime_temp_inputs: List[str] = []
        self._runtime_temp_dirs: List[str] = []
        self._runtime_unrecognized_slots = set()
        expected_clips = self._infer_expected_clips(kwargs)
        if expected_clips < 1:
            raise ValueError("No clip inputs connected. Connect at least clip1.")

        deadline = time.time() + float(wait_timeout_sec)
        clip_units: List[Dict[str, Any]] = []
        last_state = "no inputs"
        last_progress_line = ""
        signatures: Dict[str, Tuple[int, int]] = {}
        stable_counts: Dict[str, int] = {}
        while time.time() < deadline:
            clip_units, missing_slots = self._collect_clip_units(kwargs, expected_clips=expected_clips)
            if not clip_units:
                last_state = "no inputs connected yet"
                progress_line = self._build_progress_line(
                    expected_clips=expected_clips,
                    ready_slots=0,
                    missing_slots=[f"clip{i}" for i in range(1, expected_clips + 1)],
                    missing_files=[],
                    unstable_files=[],
                )
                if progress_line != last_progress_line:
                    print(progress_line)
                    last_progress_line = progress_line
                time.sleep(max(0.1, float(poll_interval_sec)))
                continue
            missing_files = self._missing_files_for_units(clip_units)
            unstable_files = self._unstable_files_for_units(
                clip_units=clip_units,
                signatures=signatures,
                stable_counts=stable_counts,
                required_stable_polls=max(1, int(stable_polls)),
            )
            ready_slots = self._count_ready_slots(
                expected_clips=expected_clips,
                clip_units=clip_units,
                missing_slots=missing_slots,
                missing_files=missing_files,
                unstable_files=unstable_files,
            )
            progress_line = self._build_progress_line(
                expected_clips=expected_clips,
                ready_slots=ready_slots,
                missing_slots=missing_slots,
                missing_files=missing_files,
                unstable_files=unstable_files,
            )
            if progress_line != last_progress_line:
                print(progress_line)
                last_progress_line = progress_line
            if ready_slots >= expected_clips:
                break
            if missing_slots:
                last_state = f"missing clip slots: {', '.join(missing_slots)}"
            elif unstable_files:
                last_state = f"waiting file-stability: {', '.join(unstable_files[:3])}"
            else:
                last_state = f"waiting for files: {', '.join(missing_files[:3])}"
            time.sleep(max(0.1, float(poll_interval_sec)))
        else:
            raise ValueError(
                f"Tektite Video Combiner 9.0 timed out after {wait_timeout_sec}s while waiting. Last state: {last_state}"
            )
        print(
            f"[Tektite Video Combiner 9.0] Ready {expected_clips}/{expected_clips}. Starting normalize+stitch."
        )

        ordered_units = clip_units

        temp_items: List[str] = []
        try:
            prepared_video_paths: List[str] = []
            for idx, unit in enumerate(ordered_units, start=1):
                if unit["kind"] == "video":
                    print(
                        f"[Tektite Video Combiner 9.0] Clip {idx}/{len(ordered_units)} is video: "
                        f"{unit.get('label', os.path.basename(unit['path']))}"
                    )
                    prepared_video_paths.append(unit["path"])
                    if os.path.basename(unit["path"]).startswith("tektite_input_"):
                        temp_items.append(unit["path"])
                else:
                    print(
                        f"[Tektite Video Combiner 9.0] Clip {idx}/{len(ordered_units)} is image sequence: "
                        f"{len(unit['paths'])} frames -> rendering video"
                    )
                    seq_video_path = self._render_image_sequence_to_video(
                        image_paths=unit["paths"],
                        fps=sequence_fps,
                        output_format=output_format,
                        overwrite=overwrite,
                        video_codec=video_codec,
                        preset=preset,
                        crf=crf,
                    )
                    prepared_video_paths.append(seq_video_path)
                    temp_items.append(seq_video_path)
                    print(
                        f"[Tektite Video Combiner 9.0] Clip {idx}/{len(ordered_units)} sequence render done: "
                        f"{seq_video_path}"
                    )

            stitched_output = self._ffmpeg_stitch(
                ordered_paths=prepared_video_paths,
                output_path=output_path,
                output_format=output_format,
                target_fps=sequence_fps,
                overwrite=overwrite,
                reencode_fallback=reencode_fallback,
                video_codec=video_codec,
                preset=preset,
                crf=crf,
            )

            if audio is not None:
                stitched_output = self._mux_audio_track(
                    video_path=stitched_output,
                    audio=audio,
                    overwrite=overwrite,
                )

            video_output = InputImpl.VideoFromFile(stitched_output)
            return (video_output, stitched_output)
        finally:
            for p in getattr(self, "_runtime_temp_inputs", []):
                if p and os.path.exists(p):
                    os.remove(p)
            for item in temp_items:
                if os.path.exists(item):
                    os.remove(item)
            for d in getattr(self, "_runtime_temp_dirs", []):
                if d and os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)

    def _build_progress_line(
        self,
        *,
        expected_clips: int,
        ready_slots: int,
        missing_slots: List[str],
        missing_files: List[str],
        unstable_files: List[str],
    ) -> str:
        ready_clips = max(0, min(int(expected_clips), int(ready_slots)))
        if missing_slots:
            state = f"waiting clips ({', '.join(missing_slots[:6])})"
        elif missing_files:
            state = "waiting files to appear"
        elif unstable_files:
            state = "stabilizing files"
        else:
            state = "ready"

        details: List[str] = []
        if missing_files:
            details.append("missing=" + ", ".join(os.path.basename(p) for p in missing_files[:2]))
        if unstable_files:
            details.append("unstable=" + ", ".join(os.path.basename(p) for p in unstable_files[:2]))
        detail_suffix = f" | {'; '.join(details)}" if details else ""

        return (
            f"[Tektite Video Combiner 9.0] Clips ready: {ready_clips}/{expected_clips} | "
            f"missing_files={len(missing_files)} unstable_files={len(unstable_files)} | {state}{detail_suffix}"
        )

    def _count_ready_slots(
        self,
        *,
        expected_clips: int,
        clip_units: List[Dict[str, Any]],
        missing_slots: List[str],
        missing_files: List[str],
        unstable_files: List[str],
    ) -> int:
        missing_slot_indexes = {
            int(s.lower().replace("clip", ""))
            for s in missing_slots
            if s.lower().startswith("clip") and s[4:].isdigit()
        }
        blocked_paths = set(missing_files) | set(unstable_files)
        slot_units: Dict[int, List[Dict[str, Any]]] = {}
        for unit in clip_units:
            slot = int(unit.get("slot", 0) or 0)
            if slot > 0:
                slot_units.setdefault(slot, []).append(unit)

        ready = 0
        for slot in range(1, int(expected_clips) + 1):
            if slot in missing_slot_indexes:
                continue
            units = slot_units.get(slot, [])
            if not units:
                continue

            slot_blocked = False
            for unit in units:
                if unit["kind"] == "video":
                    if unit["path"] in blocked_paths:
                        slot_blocked = True
                        break
                else:
                    for p in unit["paths"]:
                        if p in blocked_paths:
                            slot_blocked = True
                            break
                    if slot_blocked:
                        break

            if not slot_blocked:
                ready += 1

        return ready

    def _infer_expected_clips(self, dynamic_inputs: Dict[str, Any]) -> int:
        max_idx = 0
        pattern = re.compile(r"^clip(\d+)$", re.IGNORECASE)
        for key, value in dynamic_inputs.items():
            m = pattern.match(str(key))
            if not m:
                continue
            # Comfy can pass placeholder-ish values for wildcard slots. Only count
            # inputs that look like an actual clip source.
            if not self._has_clip_content(value):
                continue
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
        return max_idx

    def _has_clip_content(self, value: Any, _depth: int = 0, _seen: Optional[set] = None) -> bool:
        if value is None:
            return False
        if _depth > 5:
            return False
        if _seen is None:
            _seen = set()

        obj_id = id(value)
        if obj_id in _seen:
            return False
        _seen.add(obj_id)

        if self._is_image_tensor(value):
            return True
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            for key in ("fullpath", "path", "filename", "video_url", "url"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return True
            return any(self._has_clip_content(candidate, _depth + 1, _seen) for candidate in value.values())
        if isinstance(value, (tuple, list)):
            return any(self._has_clip_content(candidate, _depth + 1, _seen) for candidate in value)
        if hasattr(value, "get_stream_source") or hasattr(value, "save_to"):
            return True

        for attr in ("path", "file", "filename", "fullpath", "filepath", "file_path", "source", "stream_source"):
            candidate = getattr(value, attr, None)
            if isinstance(candidate, os.PathLike):
                candidate = os.fspath(candidate)
            if isinstance(candidate, str) and candidate.strip():
                return True
            if candidate is not None and self._has_clip_content(candidate, _depth + 1, _seen):
                return True

        for attr in ("value", "data", "video", "clip", "item", "image", "images", "frames"):
            if hasattr(value, attr):
                candidate = getattr(value, attr, None)
                if self._has_clip_content(candidate, _depth + 1, _seen):
                    return True

        return False

    def _collect_clip_units(
        self, dynamic_inputs: Dict[str, Any], *, expected_clips: int
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        units: List[Tuple[int, int, Dict[str, Any]]] = []
        pattern = re.compile(r"^clip(\d+)$", re.IGNORECASE)
        seq = 0
        per_slot_paths: Dict[int, List[str]] = {}

        for key, value in dynamic_inputs.items():
            match = pattern.match(str(key))
            if not match:
                continue

            slot_index = int(match.group(1))
            image_tensor = self._extract_image_tensor_from_value(value)
            if image_tensor is not None:
                image_paths = self._persist_image_tensor_sequence(image_tensor, slot_index=slot_index)
                if image_paths:
                    per_slot_paths.setdefault(slot_index, []).extend(image_paths)
                    units.append(
                        (
                            slot_index,
                            seq,
                            {
                                "kind": "image_sequence",
                                "paths": image_paths,
                                "label": f"clip{slot_index}_image_batch",
                                "slot": slot_index,
                                "memory_sequence": True,
                            },
                        )
                    )
                    seq += 1
                    continue

            raw_paths = self._extract_paths_from_value(value)
            paths = [self._normalize_path(p) for p in raw_paths]
            paths = self._expand_directory_sequences(paths)
            paths = [p for p in paths if p]
            if not paths:
                self._log_unrecognized_clip_input(slot_index, value)
                per_slot_paths.setdefault(slot_index, [])
                continue

            if self._all_images(paths):
                per_slot_paths.setdefault(slot_index, []).extend(paths)
                units.append(
                    (
                        slot_index,
                        seq,
                        {
                            "kind": "image_sequence",
                            "paths": self._sort_image_paths(paths),
                            "label": os.path.basename(paths[0]),
                            "slot": slot_index,
                        },
                    )
                )
                seq += 1
                continue

            recognized_paths: List[str] = []
            for p in paths:
                if self._is_video_file(p):
                    recognized_paths.append(p)
                    units.append(
                        (
                            slot_index,
                            seq,
                            {
                                "kind": "video",
                                "path": p,
                                "label": os.path.basename(p),
                                "slot": slot_index,
                            },
                        )
                    )
                    seq += 1
                elif self._is_image_file(p):
                    recognized_paths.append(p)
                    units.append(
                        (
                            slot_index,
                            seq,
                            {
                                "kind": "image_sequence",
                                "paths": [p],
                                "label": os.path.basename(p),
                                "slot": slot_index,
                            },
                        )
                    )
                    seq += 1
                elif os.path.isfile(p):
                    # Some Comfy wrappers return valid video temp files without a useful extension.
                    recognized_paths.append(p)
                    units.append(
                        (
                            slot_index,
                            seq,
                            {
                                "kind": "video",
                                "path": p,
                                "label": os.path.basename(p),
                                "slot": slot_index,
                            },
                        )
                    )
                    seq += 1

            if recognized_paths:
                per_slot_paths.setdefault(slot_index, []).extend(recognized_paths)
            else:
                per_slot_paths.setdefault(slot_index, [])

        missing_slots: List[str] = []
        for i in range(1, expected_clips + 1):
            slot_items = per_slot_paths.get(i, [])
            if not slot_items:
                missing_slots.append(f"clip{i}")

        units.sort(key=lambda item: (item[0], item[1]))
        # Keep one unit per connected input in exact slot order.
        # Do not deduplicate: repeated paths may be intentional.
        out: List[Dict[str, Any]] = [unit for _, _, unit in units]
        return out, missing_slots

    def _extract_image_tensor_from_value(self, value: Any) -> Optional[torch.Tensor]:
        if self._is_image_tensor(value):
            return value
        if isinstance(value, dict):
            for candidate in value.values():
                found = self._extract_image_tensor_from_value(candidate)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for candidate in value:
                found = self._extract_image_tensor_from_value(candidate)
                if found is not None:
                    return found
        else:
            for attr in ("image", "images", "frames", "value", "data"):
                if hasattr(value, attr):
                    found = self._extract_image_tensor_from_value(getattr(value, attr, None))
                    if found is not None:
                        return found
        return None

    def _is_image_tensor(self, value: Any) -> bool:
        if not isinstance(value, torch.Tensor):
            return False
        if value.ndim not in (3, 4):
            return False
        shape = tuple(int(x) for x in value.shape)
        if value.ndim == 3:
            return shape[-1] in (1, 3, 4) or shape[0] in (1, 3, 4)
        return shape[-1] in (1, 3, 4) or shape[1] in (1, 3, 4)

    def _persist_image_tensor_sequence(self, tensor: torch.Tensor, *, slot_index: int) -> List[str]:
        cache_key = id(tensor)
        cached = getattr(self, "_runtime_image_cache", {}).get(cache_key)
        if cached and all(os.path.exists(p) for p in cached):
            return cached

        frames = tensor.detach().float().clamp(0.0, 1.0)
        if frames.ndim == 3:
            if frames.shape[-1] in (1, 3, 4):
                frames = frames.unsqueeze(0)
            elif frames.shape[0] in (1, 3, 4):
                frames = frames.permute(1, 2, 0).unsqueeze(0)
            else:
                return []
        elif frames.ndim == 4:
            if frames.shape[-1] in (1, 3, 4):
                pass
            elif frames.shape[1] in (1, 3, 4):
                frames = frames.permute(0, 2, 3, 1)
            else:
                return []
        else:
            return []

        frame_count = int(frames.shape[0])
        if frame_count < 1:
            return []

        work_dir = tempfile.mkdtemp(prefix=f"tektite_clip{slot_index}_frames_")
        self._runtime_temp_dirs.append(work_dir)
        paths: List[str] = []
        print(
            f"[Tektite Video Combiner 9.0] clip{slot_index}: received IMAGE batch with "
            f"{frame_count} frames -> temporary image sequence"
        )
        for frame_index in range(frame_count):
            frame = (frames[frame_index].cpu().numpy() * 255.0).round().astype("uint8")
            if frame.shape[-1] == 1:
                frame = frame[:, :, 0]
            path = os.path.join(work_dir, f"frame_{frame_index:06d}.png")
            Image.fromarray(frame).save(path)
            paths.append(path)

        self._runtime_image_cache[cache_key] = paths
        return paths

    def _expand_directory_sequences(self, paths: List[str]) -> List[str]:
        expanded: List[str] = []
        for path in paths:
            if path and self._has_glob_chars(path):
                matches = [p for p in glob.glob(path) if self._is_video_file(p) or self._is_image_file(p)]
                expanded.extend(self._sort_image_paths(matches))
            elif path and os.path.isdir(path):
                try:
                    names = os.listdir(path)
                except OSError:
                    names = []
                images = [os.path.join(path, name) for name in names if self._is_image_file(name)]
                expanded.extend(self._sort_image_paths(images))
            else:
                expanded.append(path)
        return expanded

    def _missing_files_for_units(self, units: List[Dict[str, Any]]) -> List[str]:
        missing: List[str] = []
        for unit in units:
            if unit["kind"] == "video":
                path = unit["path"]
                if path.startswith(("http://", "https://")):
                    continue
                if (not os.path.exists(path)) or (os.path.getsize(path) <= 0):
                    missing.append(path)
            else:
                for path in unit["paths"]:
                    if path.startswith(("http://", "https://")):
                        continue
                    if (not os.path.exists(path)) or (os.path.getsize(path) <= 0):
                        missing.append(path)
        return missing

    def _unstable_files_for_units(
        self,
        *,
        clip_units: List[Dict[str, Any]],
        signatures: Dict[str, Tuple[int, int]],
        stable_counts: Dict[str, int],
        required_stable_polls: int,
    ) -> List[str]:
        unstable: List[str] = []
        local_paths: List[str] = []
        for unit in clip_units:
            if unit.get("memory_sequence"):
                continue
            if unit["kind"] == "video":
                path = unit["path"]
                if not path.startswith(("http://", "https://")):
                    local_paths.append(path)
            else:
                for path in unit["paths"]:
                    if not path.startswith(("http://", "https://")):
                        local_paths.append(path)

        for path in local_paths:
            if not os.path.exists(path):
                unstable.append(path)
                continue
            try:
                st = os.stat(path)
                size = int(st.st_size)
                mtime_ns = int(st.st_mtime_ns)
                sig = (size, 0)
            except OSError:
                unstable.append(path)
                continue
            # If file is non-empty and older than 2 seconds, treat it as stable even if metadata jitters.
            age_sec = max(0.0, (time.time_ns() - mtime_ns) / 1_000_000_000.0)
            if size > 0 and age_sec >= 2.0:
                signatures[path] = sig
                stable_counts[path] = max(stable_counts.get(path, 0), required_stable_polls)
                continue
            prev = signatures.get(path)
            if prev == sig:
                stable_counts[path] = stable_counts.get(path, 0) + 1
            else:
                stable_counts[path] = 0
                signatures[path] = sig
            if stable_counts[path] < (required_stable_polls - 1):
                unstable.append(path)

        return unstable

    def _extract_paths_from_value(self, value: Any, _depth: int = 0, _seen: Optional[set] = None) -> List[str]:
        paths: List[str] = []
        if _seen is None:
            _seen = set()

        if value is None:
            return paths
        if _depth > 5:
            return paths

        obj_id = id(value)
        if obj_id in _seen:
            return paths
        _seen.add(obj_id)

        if isinstance(value, os.PathLike):
            value = os.fspath(value)

        if isinstance(value, str):
            v = value.strip()
            if v:
                paths.append(v)
            return paths

        if isinstance(value, dict):
            comfy_image_path = self._path_from_comfy_image_dict(value)
            if comfy_image_path:
                return [comfy_image_path]
            for key in ("fullpath", "path", "filename", "video_url", "url"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    paths.append(candidate.strip())
            # Recursive fallback for wrapped structures.
            for key, candidate in value.items():
                if key in ("filename", "subfolder", "type"):
                    continue
                paths.extend(self._extract_paths_from_value(candidate, _depth + 1, _seen))
            return self._dedupe_keep_order(paths)

        if isinstance(value, (tuple, list)):
            # Legacy VHS shape: (something, [paths...])
            if len(value) >= 2 and isinstance(value[1], list):
                for item in value[1]:
                    paths.extend(self._extract_paths_from_value(item, _depth + 1, _seen))
            else:
                for item in value:
                    paths.extend(self._extract_paths_from_value(item, _depth + 1, _seen))
            return self._dedupe_keep_order(paths)

        # ComfyUI VIDEO object support (e.g. Load Video -> VideoFromFile).
        if hasattr(value, "get_stream_source"):
            try:
                source = value.get_stream_source()
                if isinstance(source, os.PathLike):
                    source = os.fspath(source)
                if isinstance(source, str) and source.strip():
                    return [source.strip()]

                source_paths = self._extract_paths_from_value(source, _depth + 1, _seen)
                if source_paths:
                    return source_paths

                # In-memory stream source (BytesIO, etc.): write bytes directly to temp once.
                stream_path = self._persist_stream_source_to_temp(source, owner=value)
                if stream_path:
                    return [stream_path]
            except Exception as exc:
                print(f"[Tektite Video Combiner 9.0] get_stream_source failed: {type(exc).__name__}: {exc}")

        # Fallback for video-like objects that provide save_to but no usable stream source.
        if hasattr(value, "save_to"):
            try:
                saved_path = self._persist_video_object_to_temp(value)
                if saved_path:
                    return [saved_path]
            except Exception as exc:
                print(f"[Tektite Video Combiner 9.0] save_to fallback failed: {type(exc).__name__}: {exc}")

        # Generic object fallback: common path-like attributes.
        for attr in ("path", "file", "filename", "fullpath", "filepath", "file_path", "source", "stream_source"):
            candidate = getattr(value, attr, None)
            if isinstance(candidate, os.PathLike):
                candidate = os.fspath(candidate)
            if isinstance(candidate, str) and candidate.strip():
                paths.append(candidate.strip())
            elif candidate is not None:
                paths.extend(self._extract_paths_from_value(candidate, _depth + 1, _seen))

        # Recursive object-wrapper fallback for common wrapper attrs.
        for attr in ("value", "data", "video", "clip", "item", "image", "images", "frames"):
            if hasattr(value, attr):
                candidate = getattr(value, attr, None)
                paths.extend(self._extract_paths_from_value(candidate, _depth + 1, _seen))

        return self._dedupe_keep_order(paths)

    def _path_from_comfy_image_dict(self, value: Dict[str, Any]) -> str:
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return ""
        if not self._is_image_file(filename):
            return ""

        subfolder = value.get("subfolder", "")
        if not isinstance(subfolder, str):
            subfolder = ""
        folder_type = value.get("type", "input")
        if not isinstance(folder_type, str):
            folder_type = "input"

        rel_path = os.path.join(subfolder.strip(), filename.strip()) if subfolder.strip() else filename.strip()
        if os.path.isabs(rel_path):
            return rel_path

        folder_type = folder_type.lower().strip()
        if folder_type == "output":
            base_dir = folder_paths.get_output_directory()
        elif folder_type == "temp" and hasattr(folder_paths, "get_temp_directory"):
            base_dir = folder_paths.get_temp_directory()
        else:
            base_dir = folder_paths.get_input_directory()
        return os.path.abspath(os.path.join(base_dir, rel_path))

    def _persist_stream_source_to_temp(self, source: Any, *, owner: Any) -> str:
        cache_key = (id(owner), id(source), "stream")
        cache = getattr(self, "_runtime_video_cache", {})
        cached = cache.get(cache_key)
        if cached and os.path.exists(cached):
            return cached

        data: Optional[bytes] = None
        if hasattr(source, "getbuffer"):
            try:
                data = bytes(source.getbuffer())
            except Exception:
                data = None
        if data is None and hasattr(source, "read"):
            pos = None
            try:
                if hasattr(source, "tell"):
                    pos = source.tell()
                if hasattr(source, "seek"):
                    source.seek(0)
                read_data = source.read()
                if isinstance(read_data, bytes):
                    data = read_data
                elif isinstance(read_data, str):
                    data = read_data.encode("utf-8", errors="ignore")
            finally:
                try:
                    if pos is not None and hasattr(source, "seek"):
                        source.seek(pos)
                except Exception:
                    pass

        if not data:
            return ""

        fd, tmp_path = tempfile.mkstemp(prefix="tektite_input_", suffix=".mp4")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(data)
        cache[cache_key] = tmp_path
        self._runtime_video_cache = cache
        self._runtime_temp_inputs.append(tmp_path)
        return tmp_path

    def _persist_video_object_to_temp(self, value: Any) -> str:
        cache_key = (id(value), "save_to")
        cache = getattr(self, "_runtime_video_cache", {})
        cached = cache.get(cache_key)
        if cached and os.path.exists(cached):
            return cached

        fd, tmp_path = tempfile.mkstemp(prefix="tektite_input_", suffix=".mp4")
        os.close(fd)
        value.save_to(tmp_path)
        cache[cache_key] = tmp_path
        self._runtime_video_cache = cache
        self._runtime_temp_inputs.append(tmp_path)
        return tmp_path

    def _dedupe_keep_order(self, values: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for v in values:
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def _normalize_path(self, path: Any) -> str:
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        if not isinstance(path, str):
            return ""

        candidate = os.path.expanduser(path.strip())
        if not candidate:
            return ""

        if candidate.startswith(("http://", "https://")):
            return candidate

        if self._has_glob_chars(candidate):
            if os.path.isabs(candidate):
                return candidate
            out_pattern = os.path.abspath(os.path.join(folder_paths.get_output_directory(), candidate))
            if glob.glob(out_pattern):
                return out_pattern
            inp_pattern = os.path.abspath(os.path.join(folder_paths.get_input_directory(), candidate))
            if glob.glob(inp_pattern):
                return inp_pattern
            return out_pattern

        if os.path.isabs(candidate):
            return candidate

        out = os.path.abspath(os.path.join(folder_paths.get_output_directory(), candidate))
        if os.path.exists(out):
            return out

        inp = os.path.abspath(os.path.join(folder_paths.get_input_directory(), candidate))
        if os.path.exists(inp):
            return inp

        return out

    def _is_video_file(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in VIDEO_EXTS

    def _is_image_file(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in IMAGE_EXTS

    def _has_glob_chars(self, path: str) -> bool:
        return any(ch in path for ch in ("*", "?", "["))

    def _log_unrecognized_clip_input(self, slot_index: int, value: Any) -> None:
        seen = getattr(self, "_runtime_unrecognized_slots", set())
        if slot_index in seen:
            return
        seen.add(slot_index)
        self._runtime_unrecognized_slots = seen
        print(
            f"[Tektite Video Combiner 9.0] clip{slot_index}: input connected but not recognized "
            f"as VIDEO/path/IMAGE/image-list. Debug: {self._summarize_value(value)}"
        )

    def _summarize_value(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, torch.Tensor):
            return f"torch.Tensor shape={tuple(value.shape)}"
        if isinstance(value, dict):
            keys = list(value.keys())[:8]
            return f"dict keys={keys}"
        if isinstance(value, (list, tuple)):
            first = self._summarize_value(value[0]) if value else "empty"
            return f"{type(value).__name__} len={len(value)} first={first}"
        attrs = []
        for attr in ("path", "filename", "subfolder", "type", "value", "data", "video", "image", "images"):
            if hasattr(value, attr):
                attrs.append(attr)
        return f"{type(value).__module__}.{type(value).__name__} attrs={attrs[:8]}"

    def _all_images(self, paths: List[str]) -> bool:
        return len(paths) > 0 and all(self._is_image_file(p) for p in paths)

    def _sort_image_paths(self, paths: List[str]) -> List[str]:
        def key_fn(p: str):
            name = os.path.basename(p)
            chunks = re.split(r"(\d+)", name)
            out = []
            for c in chunks:
                out.append(int(c) if c.isdigit() else c.lower())
            return out

        return sorted(paths, key=key_fn)

    def _render_image_sequence_to_video(
        self,
        *,
        image_paths: List[str],
        fps: float,
        output_format: str,
        overwrite: bool,
        video_codec: str,
        preset: str,
        crf: int,
    ) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to encode image sequences.")
        if not image_paths:
            raise ValueError("Empty image sequence received.")

        work_dir = tempfile.mkdtemp(prefix="tektite_seq_")
        self._runtime_temp_dirs.append(work_dir)
        list_path = os.path.join(work_dir, "images.txt")
        out_path = os.path.join(work_dir, f"sequence.{output_format}")
        print(
            f"[Tektite Video Combiner 9.0] Rendering image sequence: "
            f"{len(image_paths)} frames at {float(fps):.2f} fps -> {out_path}"
        )

        staged_pattern = os.path.join(work_dir, "frame_%06d.png")
        for frame_index, source_path in enumerate(image_paths, start=1):
            if not os.path.exists(source_path):
                raise RuntimeError(f"Image sequence frame is missing before encode: {source_path}")
            staged_path = os.path.join(work_dir, f"frame_{frame_index:06d}.png")
            try:
                with Image.open(source_path) as image:
                    image.convert("RGB").save(staged_path)
            except Exception as exc:
                raise RuntimeError(f"Could not read image sequence frame: {source_path} ({exc})") from exc

        overwrite_flag = "-y" if overwrite else "-n"
        cmd = [
            ffmpeg_path,
            overwrite_flag,
            "-framerate",
            str(max(float(fps), 1.0)),
            "-i",
            staged_pattern,
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        cmd = [str(x) for x in cmd]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg image-sequence encode failed.\n{result.stderr.strip()}")

        print(f"[Tektite Video Combiner 9.0] Image sequence video ready: {out_path}")
        return out_path

    def _ffmpeg_stitch(
        self,
        *,
        ordered_paths: List[str],
        output_path: str,
        output_format: str,
        target_fps: float,
        overwrite: bool,
        reencode_fallback: bool,
        video_codec: str,
        preset: str,
        crf: int,
    ) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to enable stitching.")

        resolved_output = self._resolve_output_path(output_path=output_path, output_format=output_format)
        resolved_output = self._ensure_writable_output_path(resolved_output)
        os.makedirs(os.path.dirname(resolved_output), exist_ok=True)

        concat_list_path = ""
        normalized_dir = tempfile.mkdtemp(prefix="tektite_norm_")
        normalized_paths: List[str] = []
        try:
            # Stage 1: normalize each clip to stable CFR intermediates first.
            for idx, source in enumerate(ordered_paths):
                norm_path = os.path.join(normalized_dir, f"norm_{idx:05d}.mp4")
                self._normalize_clip_for_concat(
                    source_path=source,
                    out_path=norm_path,
                    target_fps=target_fps,
                    overwrite=overwrite,
                    video_codec=video_codec,
                    preset=preset,
                    crf=crf,
                )
                normalized_paths.append(norm_path)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as c:
                concat_list_path = c.name
                for p in normalized_paths:
                    ep = p.replace("\\", "\\\\").replace("'", r"'\\''")
                    c.write(f"file '{ep}'\n")

            overwrite_flag = "-y" if overwrite else "-n"
            # Always re-encode to normalize timestamps/timebase and avoid freeze/black-frame artifacts.
            reencode_cmd = [
                ffmpeg_path,
                overwrite_flag,
                "-f",
                "concat",
                "-safe",
                "0",
                "-fflags",
                "+genpts",
                "-i",
                concat_list_path,
                "-fps_mode",
                "cfr",
                "-r",
                str(float(target_fps)),
                "-map",
                "0:v:0",
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                resolved_output,
            ]
            reencode_cmd = [str(x) for x in reencode_cmd]
            reencode_result = subprocess.run(reencode_cmd, capture_output=True, text=True)
            if reencode_result.returncode != 0:
                raise RuntimeError(
                    "FFmpeg stitch failed.\n"
                    f"{reencode_result.stderr.strip()}"
                )

            return resolved_output
        finally:
            if concat_list_path and os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            if os.path.isdir(normalized_dir):
                shutil.rmtree(normalized_dir, ignore_errors=True)

    def _normalize_clip_for_concat(
        self,
        *,
        source_path: str,
        out_path: str,
        target_fps: float,
        overwrite: bool,
        video_codec: str,
        preset: str,
        crf: int,
    ) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to normalize clips.")

        overwrite_flag = "-y" if overwrite else "-n"
        gop = max(1, int(round(float(target_fps) * 2.0)))
        cmd = [
            ffmpeg_path,
            overwrite_flag,
            "-fflags",
            "+genpts",
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"fps={float(target_fps)},format=yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            str(float(target_fps)),
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            out_path,
        ]
        cmd = [str(x) for x in cmd]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                "FFmpeg clip normalization failed.\n"
                f"Source: {source_path}\n"
                f"{res.stderr.strip()}"
            )

    def _mux_audio_track(self, *, video_path: str, audio: Any, overwrite: bool) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to mux audio.")

        wav_path = self._write_audio_temp_wav(audio)
        if not wav_path:
            return video_path

        root, ext = os.path.splitext(video_path)
        muxed_path = f"{root}_muxed{ext}"
        overwrite_flag = "-y" if overwrite else "-n"

        try:
            cmd = [
                ffmpeg_path,
                overwrite_flag,
                "-i",
                video_path,
                "-i",
                wav_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                muxed_path,
            ]
            cmd = [str(x) for x in cmd]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg audio mux failed.\n{result.stderr.strip()}")

            os.replace(muxed_path, video_path)
            return video_path
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(muxed_path):
                os.remove(muxed_path)

    def _write_audio_temp_wav(self, audio: Any) -> str:
        if not isinstance(audio, dict):
            return ""
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0) or 0)
        if waveform is None or sample_rate <= 0:
            return ""
        if not isinstance(waveform, torch.Tensor):
            return ""

        # Expected shape: [batch, channels, samples] or [channels, samples]
        if waveform.ndim == 3:
            data = waveform[0]
        elif waveform.ndim == 2:
            data = waveform
        elif waveform.ndim == 1:
            data = waveform.unsqueeze(0)
        else:
            return ""

        if data.shape[0] > 2:
            data = data[:2, :]

        pcm = (
            data.clamp(-1.0, 1.0)
            .mul(32768.0)
            .to(torch.int16)
            .transpose(0, 1)
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
        )

        fd, wav_path = tempfile.mkstemp(prefix="tektite_audio_", suffix=".wav")
        os.close(fd)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(int(data.shape[0]))
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return wav_path

    def _resolve_output_path(self, *, output_path: str, output_format: str) -> str:
        base_output_dir = folder_paths.get_output_directory()
        stamp = time.strftime("%Y%m%d_%H%M%S")

        raw = (output_path or "").strip()
        if not raw:
            raw = os.path.join("tektite", "stitched", f"stitched_{stamp}.{output_format}")

        expanded = os.path.expanduser(raw)
        if not os.path.isabs(expanded):
            expanded = os.path.join(base_output_dir, expanded)

        if expanded.endswith(os.sep) or os.path.isdir(expanded):
            expanded = os.path.join(expanded, f"stitched_{stamp}.{output_format}")

        _, ext = os.path.splitext(expanded)
        if not ext:
            expanded = f"{expanded}.{output_format}"

        return os.path.abspath(expanded)

    def _ensure_writable_output_path(self, resolved_output: str) -> str:
        out_dir = os.path.dirname(resolved_output) or folder_paths.get_output_directory()
        if os.path.isdir(out_dir) and os.access(out_dir, os.W_OK):
            return resolved_output

        fallback_dir = folder_paths.get_output_directory()
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, os.path.basename(resolved_output))
        print(
            f"[Tektite Video Combiner 9.0] Output directory not writable: {out_dir}. "
            f"Falling back to: {fallback_path}"
        )
        return fallback_path


NODE_CLASS_MAPPINGS = {
    "TektiteVideoCombiner9": TektiteVideoCombiner9,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TektiteVideoCombiner9": "Tektite Video Combiner 9.0",
}
