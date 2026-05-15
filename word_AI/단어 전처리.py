import csv
import os
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import config

POSE_LEN = 25 * 3
HAND_LEN = 21 * 3
FACE_LEN = 70 * 3

# Progress logging intervals (tune if needed)
SCAN_LOG_INTERVAL = 10000
DIR_LOG_INTERVAL = 200
VIDEO_LOG_INTERVAL = 200

FILENAME_RE = re.compile(r"(.+?)_(\d+)_keypoints\.json$")
LABEL_RE = re.compile(r"(WORD\d+|FS\d+)")


@dataclass
class FrameRef:
    frame_num: int
    zip_path: Path
    member_name: str


@dataclass
class VideoMeta:
    video_uid: str
    video_id: str
    label: str
    source_zip: str


def _output(path_name: str) -> Path:
    return config.PREPROCESS_OUTPUT_DIR / path_name


def _is_front_video(video_id: str) -> bool:
    tokens = video_id.split("_")
    return "F" in tokens


def parse_filename(member_name: str, front_only: bool = True):
    filename = Path(member_name).name
    m = FILENAME_RE.match(filename)
    if m is None:
        return None

    video_id = m.group(1)
    frame_num = int(m.group(2))

    lm = LABEL_RE.search(filename)
    if lm is None:
        return None
    label = lm.group(1)

    if front_only and not _is_front_video(video_id):
        return None

    return video_id, frame_num, label


def get_people(data: dict) -> dict:
    people = data.get("people", {})
    if isinstance(people, list):
        if not people:
            return {}
        return people[0] if isinstance(people[0], dict) else {}
    if isinstance(people, dict):
        return people
    return {}


def fix_length(arr, target_len: int) -> list[float]:
    if not isinstance(arr, list):
        return [0.0] * target_len
    if len(arr) >= target_len:
        return arr[:target_len]
    return arr + [0.0] * (target_len - len(arr))


def reshape_keypoints(flat_arr: list[float]) -> np.ndarray:
    return np.asarray(flat_arr, dtype=np.float32).reshape(-1, 3)


def get_origin_and_scale(pose_arr: np.ndarray):
    origin = np.array([0.0, 0.0], dtype=np.float32)
    scale = 1.0

    try:
        neck = pose_arr[1]
        r_shoulder = pose_arr[2]
        l_shoulder = pose_arr[5]

        if r_shoulder[2] > 0 and l_shoulder[2] > 0:
            origin = (r_shoulder[:2] + l_shoulder[:2]) / 2.0
            shoulder_width = float(np.linalg.norm(r_shoulder[:2] - l_shoulder[:2]))
            if shoulder_width > 1e-6:
                scale = shoulder_width
        elif neck[2] > 0:
            origin = neck[:2].astype(np.float32)
    except Exception:
        pass

    return origin, scale


def normalize_points(arr: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    out = arr.copy()
    confident = out[:, 2] > 0
    out[~confident, 0:2] = 0.0
    out[confident, 0] = (out[confident, 0] - origin[0]) / scale
    out[confident, 1] = (out[confident, 1] - origin[1]) / scale
    return out


def extract_frame_feature(data: dict) -> np.ndarray:
    people = get_people(data)

    pose = reshape_keypoints(fix_length(people.get("pose_keypoints_2d", []), POSE_LEN))
    left = reshape_keypoints(fix_length(people.get("hand_left_keypoints_2d", []), HAND_LEN))
    right = reshape_keypoints(fix_length(people.get("hand_right_keypoints_2d", []), HAND_LEN))

    origin, scale = get_origin_and_scale(pose)

    pose = normalize_points(pose, origin, scale)
    left = normalize_points(left, origin, scale)
    right = normalize_points(right, origin, scale)

    features = [pose.ravel(), left.ravel(), right.ravel()]

    if config.USE_FACE:
        face = reshape_keypoints(fix_length(people.get("face_keypoints_2d", []), FACE_LEN))
        face = normalize_points(face, origin, scale)
        features.append(face.ravel())

    return np.concatenate(features).astype(np.float32)


def resize_sequence(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    current_len = len(sequence)
    if current_len == seq_len:
        return sequence
    if current_len > seq_len:
        idx = np.linspace(0, current_len - 1, seq_len).astype(int)
        return sequence[idx]

    feature_dim = sequence.shape[1]
    padding = np.zeros((seq_len - current_len, feature_dim), dtype=np.float32)
    return np.vstack([sequence, padding])


def maybe_load_morpheme_map() -> dict[str, str]:
    morpheme_zip = config.resolve_morpheme_zip_path()
    if not morpheme_zip.exists():
        return {}

    result: dict[str, str] = {}
    try:
        with ZipFile(morpheme_zip, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".json")]
            for member in members:
                with zf.open(member) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    items = [data]
                elif isinstance(data, list):
                    items = data
                else:
                    continue

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = json.dumps(item, ensure_ascii=False)
                    lm = LABEL_RE.search(text)
                    if lm is None:
                        continue
                    label = lm.group(1)
                    word_name = None
                    for k in ["word", "WORD", "label", "gloss", "meaning"]:
                        if k in item and isinstance(item[k], str):
                            word_name = item[k]
                            break
                    if word_name:
                        result[label] = word_name
    except Exception:
        return {}

    return result


def _can_use_cache(zip_paths: list[Path]) -> bool:
    required = [
        _output(config.X_FILENAME),
        _output(config.Y_FILENAME),
        _output(config.LABEL_MAP_FILENAME),
        _output(config.LABEL_COUNTS_FILENAME),
        _output(config.USED_VIDEOS_FILENAME),
        _output(config.PREPROCESS_META_FILENAME),
    ]
    if not all(p.exists() for p in required):
        return False

    try:
        meta = json.loads(_output(config.PREPROCESS_META_FILENAME).read_text(encoding="utf-8"))
    except Exception:
        return False

    current = {
        "seq_len": config.SEQ_LEN,
        "use_face": config.USE_FACE,
        "front_only": config.FRONT_ONLY,
        "train_mode": config.TRAIN_MODE,
        "use_all_labels": config.USE_ALL_LABELS,
        "top_n_labels": config.TOP_N_LABELS,
        "zip_files": [str(p) for p in zip_paths],
    }
    return meta == current


def preprocess_from_folder() -> tuple[dict, dict, list, int]:
    """Scan pre-extracted folder of keypoint JSONs (for folder mode)."""
    if not config.EXTRACTED_KEYPOINTS_FOLDER.exists():
        raise FileNotFoundError(f"Extracted keypoints folder not found: {config.EXTRACTED_KEYPOINTS_FOLDER}")
    
    print(f"[preprocess] scanning extracted folder: {config.EXTRACTED_KEYPOINTS_FOLDER}")
    scan_started_at = time.time()
    
    video_frames: dict[str, list[Path]] = defaultdict(list)
    video_meta: dict[str, VideoMeta] = {}
    errors: list[dict] = []
    total_json_files = 0
    total_dirs = 0
    
    for root, dirnames, filenames in os.walk(config.EXTRACTED_KEYPOINTS_FOLDER):
        total_dirs += 1
        dirnames.sort()
        filenames = sorted(filenames)

        if total_dirs % DIR_LOG_INTERVAL == 0:
            elapsed = time.time() - scan_started_at
            print(
                f"[preprocess] folder scan progress: dirs={total_dirs:,}, "
                f"json={total_json_files:,}, videos={len(video_frames):,}, elapsed={elapsed:.1f}s"
            )

        root_path = Path(root)

        for filename in filenames:
            if not filename.endswith("_keypoints.json"):
                continue

            json_path = root_path / filename
            total_json_files += 1
            member_name = str(json_path.relative_to(config.EXTRACTED_KEYPOINTS_FOLDER))
            parsed = parse_filename(member_name, front_only=config.FRONT_ONLY)
            if parsed is None:
                continue
            
            video_id, frame_num, label = parsed
            video_uid = video_id
            video_frames[video_uid].append((frame_num, json_path))
            
            if video_uid not in video_meta:
                video_meta[video_uid] = VideoMeta(
                    video_uid=video_uid,
                    video_id=video_id,
                    label=label,
                    source_zip="extracted_folder",
                )

            if total_json_files % SCAN_LOG_INTERVAL == 0:
                elapsed = time.time() - scan_started_at
                print(
                    f"[preprocess] folder scan progress: dirs={total_dirs:,}, "
                    f"json={total_json_files:,}, videos={len(video_frames):,}, elapsed={elapsed:.1f}s"
                )
    
    elapsed = time.time() - scan_started_at
    print(f"[preprocess] folder scan complete: dirs={total_dirs:,}, json={total_json_files:,}, elapsed={elapsed:.1f}s")
    print(f"[preprocess] total keypoint json count: {total_json_files}")
    print(f"[preprocess] total video samples: {len(video_frames)}")
    
    return video_frames, video_meta, errors, total_json_files


def preprocess(force: bool = False) -> dict:
    config.ensure_dirs()
    
    # Choose source mode
    if config.PREPROCESS_SOURCE_MODE == "folder":
        print("[preprocess] mode: folder (pre-extracted JSONs)")
        video_frames, video_meta, errors, total_json_files = preprocess_from_folder()
        source_info = {"mode": "folder", "folder": str(config.EXTRACTED_KEYPOINTS_FOLDER)}
    else:
        print("[preprocess] mode: zip (streaming)")
        zip_paths = config.resolve_zip_paths()
        print("[preprocess] zip files to use:")
        for p in zip_paths:
            print(" -", p)
        
        missing = [str(p) for p in zip_paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing zip files:\n" + "\n".join(missing))
        
        use_cache = (not force) and (not config.FORCE_PREPROCESS) and _can_use_cache(zip_paths)
        if use_cache:
            print("[preprocess] cache hit: using existing npy/csv/json outputs")
            X = np.load(_output(config.X_FILENAME), mmap_mode="r")
            y = np.load(_output(config.Y_FILENAME), mmap_mode="r")
            return {
                "cached": True,
                "x_shape": tuple(X.shape),
                "y_shape": tuple(y.shape),
                "zip_files": [str(p) for p in zip_paths],
            }
        
        print("[preprocess] scanning zip members...")
        scan_started_at = time.time()
        video_frames: dict[str, list[FrameRef]] = defaultdict(list)
        video_meta: dict[str, VideoMeta] = {}
        errors: list[dict] = []
        total_json_files = 0
        
        for zip_path in zip_paths:
            try:
                with ZipFile(zip_path, "r") as zf:
                    for member in zf.namelist():
                        if not member.endswith("_keypoints.json"):
                            continue
                        total_json_files += 1
                        parsed = parse_filename(member, front_only=config.FRONT_ONLY)
                        if parsed is None:
                            continue
                        video_id, frame_num, label = parsed
                        video_uid = f"{zip_path.name}::{video_id}"
                        video_frames[video_uid].append(FrameRef(frame_num=frame_num, zip_path=zip_path, member_name=member))
                        if video_uid not in video_meta:
                            video_meta[video_uid] = VideoMeta(
                                video_uid=video_uid,
                                video_id=video_id,
                                label=label,
                                source_zip=zip_path.name,
                            )

                        if total_json_files % SCAN_LOG_INTERVAL == 0:
                            elapsed = time.time() - scan_started_at
                            print(
                                f"[preprocess] zip scan progress: {total_json_files:,} json files, "
                                f"{len(video_frames):,} videos, elapsed {elapsed:.1f}s"
                            )
            except Exception as e:
                errors.append({
                    "zip_path": str(zip_path),
                    "video_uid": "",
                    "member": "",
                    "stage": "scan_zip",
                    "error": str(e),
                })
        
        print(f"[preprocess] total keypoint json count: {total_json_files}")
        print(f"[preprocess] total video samples before label filter: {len(video_frames)}")
        source_info = {"mode": "zip", "zip_files": [str(p) for p in zip_paths]}

    label_counter = Counter(v.label for v in video_meta.values())

    if config.TRAIN_MODE == "debug":
        top_n = config.TOP_N_LABELS if config.TOP_N_LABELS is not None else config.DEBUG_TOP_N_LABELS
        selected_labels = [lbl for lbl, _ in label_counter.most_common(top_n)]
    else:
        if config.USE_ALL_LABELS and config.TOP_N_LABELS is None:
            selected_labels = sorted(label_counter.keys())
        elif config.TOP_N_LABELS is not None:
            selected_labels = [lbl for lbl, _ in label_counter.most_common(config.TOP_N_LABELS)]
        else:
            selected_labels = sorted(label_counter.keys())

    selected_label_set = set(selected_labels)
    print(f"[preprocess] selected labels: {len(selected_labels)}")

    # Build label map first to keep y integer-encoded.
    label_to_idx = {label: idx for idx, label in enumerate(selected_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    feature_dim = 201 + (FACE_LEN if config.USE_FACE else 0)
    selected_video_ids = [uid for uid, m in video_meta.items() if m.label in selected_label_set]

    if config.TRAIN_MODE == "debug":
        selected_video_ids = selected_video_ids[: config.DEBUG_MAX_VIDEOS]

    print(f"[preprocess] selected video samples: {len(selected_video_ids)}")

    if not selected_video_ids:
        raise ValueError("No video samples selected. Check FRONT_ONLY/TRAIN_MODE/filter options.")

    # Optional memmap to reduce RAM pressure.
    X_mem = None
    y_mem = None
    if config.USE_MEMMAP:
        x_mem_path = _output(config.MEMMAP_FILENAME)
        y_mem_path = _output("y_memmap.dat")
        X_mem = np.memmap(x_mem_path, dtype=np.float32, mode="w+", shape=(len(selected_video_ids), config.SEQ_LEN, feature_dim))
        y_mem = np.memmap(y_mem_path, dtype=np.int32, mode="w+", shape=(len(selected_video_ids),))

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    used_rows: list[dict] = []

    open_zips: dict[str, ZipFile] = {}
    written = 0

    try:
        # Timing / ETA helpers
        total_videos = len(selected_video_ids)
        overall_start_time = time.time()
        last_log_time = overall_start_time
        last_log_index = 0

        def _fmt_sec(s: float) -> str:
            return time.strftime('%H:%M:%S', time.gmtime(int(s)))

        for i, video_uid in enumerate(selected_video_ids, start=1):
            meta = video_meta[video_uid]
            frame_refs = sorted(video_frames[video_uid], key=lambda x: x[0] if isinstance(x, tuple) else x.frame_num)

            frame_features = []
            for ref in frame_refs:
                try:
                    if config.PREPROCESS_SOURCE_MODE == "folder":
                        # Folder mode: ref is (frame_num, json_path)
                        frame_num, json_path = ref
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        # ZIP mode: ref is FrameRef
                        zf = open_zips.get(str(ref.zip_path))
                        if zf is None:
                            zf = ZipFile(ref.zip_path, "r")
                            open_zips[str(ref.zip_path)] = zf
                        with zf.open(ref.member_name) as f:
                            data = json.load(f)
                    
                    frame_features.append(extract_frame_feature(data))
                except Exception as e:
                    if config.PREPROCESS_SOURCE_MODE == "folder":
                        frame_num, json_path = ref
                        errors.append({
                            "source": "folder",
                            "json_path": str(json_path),
                            "video_uid": video_uid,
                            "stage": "read_folder",
                            "error": str(e),
                        })
                    else:
                        errors.append({
                            "zip_path": str(ref.zip_path),
                            "video_uid": video_uid,
                            "member": ref.member_name,
                            "stage": "read_or_extract",
                            "error": str(e),
                        })
                    continue

            if len(frame_features) < config.MIN_VIDEO_FRAMES:
                continue

            sequence = np.asarray(frame_features, dtype=np.float32)
            sequence = resize_sequence(sequence, config.SEQ_LEN)
            y_idx = label_to_idx[meta.label]

            if config.USE_MEMMAP:
                X_mem[written] = sequence
                y_mem[written] = y_idx
            else:
                X_list.append(sequence)
                y_list.append(y_idx)

            used_rows.append({
                "video_uid": meta.video_uid,
                "video_id": meta.video_id,
                "label": meta.label,
                "label_idx": y_idx,
                "source_zip": meta.source_zip,
                "num_frames_raw": len(frame_refs),
            })
            written += 1

            if i % VIDEO_LOG_INTERVAL == 0:
                now = time.time()
                chunk_count = i - last_log_index
                chunk_time = now - last_log_time if last_log_time is not None else 0.0
                per_video = (chunk_time / chunk_count) if chunk_count > 0 else 0.0
                processed = i
                remaining = max(0, total_videos - processed)
                est_remaining = remaining * per_video
                elapsed_total = now - overall_start_time
                est_total = elapsed_total + est_remaining

                print(
                    f"[preprocess] processed videos: {processed}/{total_videos} | "
                    f"chunk_time={chunk_time:.1f}s ({chunk_count} videos) | "
                    f"per_video={per_video:.3f}s | elapsed={_fmt_sec(elapsed_total)} | "
                    f"eta_remaining={_fmt_sec(est_remaining)} | eta_finish={_fmt_sec(est_total)}"
                )

                last_log_time = now
                last_log_index = i
    finally:
        # Close ZIP handles (folder mode doesn't use them)
        if config.PREPROCESS_SOURCE_MODE == "zip":
            for zf in open_zips.values():
                zf.close()

    if written == 0:
        raise ValueError("No valid sequences built. Check parsing rules and JSON structure.")

    if config.USE_MEMMAP:
        X = np.asarray(X_mem[:written], dtype=np.float32)
        y = np.asarray(y_mem[:written], dtype=np.int64)
    else:
        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int64)

    one_frame = X[0, 0]
    if one_frame.shape[0] != feature_dim:
        raise ValueError(f"Feature dimension mismatch: expected {feature_dim}, got {one_frame.shape[0]}")

    np.save(_output(config.X_FILENAME), X)
    np.save(_output(config.Y_FILENAME), y)

    used_df = pd.DataFrame(used_rows)
    used_df.to_csv(_output(config.USED_VIDEOS_FILENAME), index=False, encoding="utf-8-sig")

    label_counts = used_df.groupby("label").size().sort_values(ascending=False).rename("sample_count").reset_index()
    label_counts.to_csv(_output(config.LABEL_COUNTS_FILENAME), index=False, encoding="utf-8-sig")

    morpheme_map = maybe_load_morpheme_map()
    label_map_payload = {
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "label_to_idx": label_to_idx,
        "label_to_word": {k: morpheme_map.get(k, "") for k in selected_labels},
    }
    _output(config.LABEL_MAP_FILENAME).write_text(
        json.dumps(label_map_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(errors).to_csv(_output(config.ERROR_LOG_FILENAME), index=False, encoding="utf-8-sig")

    meta = {
        "seq_len": config.SEQ_LEN,
        "use_face": config.USE_FACE,
        "front_only": config.FRONT_ONLY,
        "train_mode": config.TRAIN_MODE,
        "use_all_labels": config.USE_ALL_LABELS,
        "top_n_labels": config.TOP_N_LABELS,
        "source_info": source_info,
    }
    _output(config.PREPROCESS_META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[preprocess] used zip list:")
    for p in zip_paths:
        print(" -", p)
    print("[preprocess] total json files:", total_json_files)
    print("[preprocess] final sample count:", X.shape[0])
    print("[preprocess] used label count:", len(set(y.tolist())))
    print("[preprocess] one-frame feature shape:", one_frame.shape)
    print("[preprocess] final X shape:", X.shape)
    print("[preprocess] final y shape:", y.shape)

    return {
        "cached": False,
        "zip_files": [str(p) for p in zip_paths],
        "total_json_files": total_json_files,
        "sample_count": int(X.shape[0]),
        "label_count": int(len(set(y.tolist()))),
        "one_frame_feature_shape": tuple(one_frame.shape),
        "x_shape": tuple(X.shape),
        "y_shape": tuple(y.shape),
    }


if __name__ == "__main__":
    preprocess(force=False)
