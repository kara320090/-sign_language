"""
build_sentence_dataset_from_aihub.py

AIHub 수어 keypoint F 폴더를 직접 읽어서 sen_AI 학습/검증용 데이터를 생성한다.

생성 파일:
    sen_AI/data/processed/X.npy
    sen_AI/data/processed/y.npy
    sen_AI/data/processed/classes.npy
    sen_AI/data/processed/label_map.json
    sen_AI/data/processed/preprocess_meta.csv
    sen_AI/data/processed/preprocess_report.json

    sen_AI/data/train/X_train.npy
    sen_AI/data/train/y_train.npy
    sen_AI/data/validation/X_validation.npy
    sen_AI/data/validation/y_validation.npy
    sen_AI/data/validation/classes.npy

입력 구조 예시:
    AIHub_F_keypoints/
    ├─ NIA_SL_SEN2000_REAL05_F/
    │  ├─ NIA_SL_SEN2000_REAL05_F_000000000000_keypoints.json
    │  ├─ NIA_SL_SEN2000_REAL05_F_000000000001_keypoints.json
    │  └─ ...
    ├─ NIA_SL_SEN1999_REAL05_F/
    │  └─ ...

기본 feature:
    left hand  21점 x,y = 42
    right hand 21점 x,y = 42
    pose       18점 x,y = 36
    총 120D

실행 예:
    cd sen_AI
    python data/build_sentence_dataset_from_aihub.py ^
      --input "D:/AIHub/front_keypoints" ^
      --metadata "D:/AIHub/sentence_labels.csv" ^
      --label-col label ^
      --id-col sample_id

metadata가 없으면 폴더명에서 SEN번호를 라벨로 사용한다.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEQ_LEN = 30
FEATURE_DIM = 120
RANDOM_SEED = 42

# pose 18개 x,y = 36D
# OpenPose BODY_25 기준 앞 18개를 기본 사용
POSE_18_INDICES = list(range(18))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_frame_number(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return -1
    return int(nums[-1])


def parse_sen_id(name: str) -> str:
    match = re.search(r"(SEN\d+)", name)
    if not match:
        return name
    return match.group(1)


def find_sequence_dirs(input_dir: Path) -> List[Path]:
    """
    F 폴더만 사용한다.
    폴더명 예:
        NIA_SL_SEN2000_REAL05_F
    """
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    sequence_dirs = []

    for p in input_dir.rglob("*"):
        if not p.is_dir():
            continue

        if not p.name.endswith("_F"):
            continue

        json_files = list(p.glob("*.json"))
        keypoint_files = [
            j for j in json_files
            if "keypoint" in j.name.lower() or "keypoints" in j.name.lower()
        ]

        if keypoint_files:
            sequence_dirs.append(p)

    sequence_dirs = sorted(sequence_dirs, key=lambda x: x.name)
    return sequence_dirs


def recursive_find_key(obj: Any, key: str) -> Optional[list]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, key)
            if found is not None:
                return found

    return None


def get_people_data(data: dict) -> Any:
    people = data.get("people")

    if isinstance(people, list) and len(people) > 0:
        # 여러 명이 있으면 평균 confidence가 가장 높은 사람 선택
        best = people[0]
        best_score = -1.0

        for person in people:
            if not isinstance(person, dict):
                continue

            score_values = []

            for key in [
                "hand_left_keypoints_2d",
                "hand_right_keypoints_2d",
                "pose_keypoints_2d",
            ]:
                values = person.get(key)
                if values is None:
                    continue

                arr = np.asarray(values, dtype=np.float32).flatten()
                if len(arr) % 3 == 0 and len(arr) > 0:
                    score_values.append(float(np.mean(arr.reshape(-1, 3)[:, 2])))

            score = float(np.mean(score_values)) if score_values else 0.0

            if score > best_score:
                best_score = score
                best = person

        return best

    if isinstance(people, dict):
        return people

    return data


def keypoints_to_xy(
    values: Optional[list],
    num_points: int,
    width: float = 1920.0,
    height: float = 1080.0,
    conf_threshold: float = 0.05,
) -> np.ndarray:
    """
    keypoint list를 (num_points, 2)로 변환한다.
    입력이 x,y,confidence 반복이면 confidence가 낮은 점은 0 처리.
    입력이 x,y 반복이어도 처리.
    """
    out = np.zeros((num_points, 2), dtype=np.float32)

    if values is None:
        return out

    arr = np.asarray(values, dtype=np.float32).flatten()

    if len(arr) == 0:
        return out

    if len(arr) % 3 == 0:
        arr3 = arr.reshape(-1, 3)
        xy = arr3[:, :2]
        conf = arr3[:, 2]
    elif len(arr) % 2 == 0:
        xy = arr.reshape(-1, 2)
        conf = np.ones((len(xy),), dtype=np.float32)
    else:
        return out

    n = min(num_points, len(xy))
    xy = xy[:n].astype(np.float32)
    conf = conf[:n].astype(np.float32)

    # 좌표가 pixel 값이면 0~1로 정규화
    # 이미 0~1 범위면 그대로 둔다.
    if np.nanmax(xy) > 2.0:
        xy[:, 0] = xy[:, 0] / width
        xy[:, 1] = xy[:, 1] / height

    xy[conf < conf_threshold] = 0.0
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    xy = np.clip(xy, 0.0, 1.0)

    out[:n] = xy
    return out


def extract_frame_feature(json_path: Path) -> np.ndarray:
    """
    1프레임에서 120D feature 추출.

    left hand  21*2 = 42
    right hand 21*2 = 42
    pose       18*2 = 36
    total           = 120
    """
    data = load_json(json_path)
    person = get_people_data(data)

    left_values = None
    right_values = None
    pose_values = None

    if isinstance(person, dict):
        left_values = person.get("hand_left_keypoints_2d")
        right_values = person.get("hand_right_keypoints_2d")
        pose_values = person.get("pose_keypoints_2d")

    if left_values is None:
        left_values = recursive_find_key(data, "hand_left_keypoints_2d")
    if right_values is None:
        right_values = recursive_find_key(data, "hand_right_keypoints_2d")
    if pose_values is None:
        pose_values = recursive_find_key(data, "pose_keypoints_2d")

    left_xy = keypoints_to_xy(left_values, num_points=21)
    right_xy = keypoints_to_xy(right_values, num_points=21)

    pose_all = keypoints_to_xy(pose_values, num_points=25)
    pose_18 = np.zeros((18, 2), dtype=np.float32)

    for out_idx, pose_idx in enumerate(POSE_18_INDICES):
        if pose_idx < len(pose_all):
            pose_18[out_idx] = pose_all[pose_idx]

    feature = np.concatenate(
        [
            left_xy.flatten(),
            right_xy.flatten(),
            pose_18.flatten(),
        ],
        axis=0,
    ).astype(np.float32)

    if feature.shape[0] != FEATURE_DIM:
        raise ValueError(f"feature dim error: {feature.shape[0]} != {FEATURE_DIM}")

    return feature


def fix_sequence_length(sequence: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """
    sequence shape: (T, F)
    길면 균등 샘플링, 짧으면 마지막 프레임 반복 padding.
    """
    if sequence.ndim != 2:
        raise ValueError(f"sequence must be 2D, got {sequence.shape}")

    t = sequence.shape[0]

    if t == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)

    if t == target_len:
        return sequence.astype(np.float32)

    if t > target_len:
        indices = np.linspace(0, t - 1, target_len).astype(int)
        return sequence[indices].astype(np.float32)

    pad_count = target_len - t
    last = sequence[-1:]
    padding = np.repeat(last, pad_count, axis=0)
    return np.concatenate([sequence, padding], axis=0).astype(np.float32)


def load_sequence(seq_dir: Path) -> Tuple[np.ndarray, int, int]:
    json_files = [
        p for p in seq_dir.glob("*.json")
        if "keypoint" in p.name.lower() or "keypoints" in p.name.lower()
    ]

    json_files = sorted(json_files, key=extract_frame_number)

    if not json_files:
        raise FileNotFoundError(f"no keypoint json files: {seq_dir}")

    features = []

    for json_path in json_files:
        feature = extract_frame_feature(json_path)
        features.append(feature)

    raw = np.stack(features, axis=0).astype(np.float32)
    fixed = fix_sequence_length(raw, SEQ_LEN)

    return fixed, len(json_files), int(np.count_nonzero(raw))


def sample_hash(x: np.ndarray) -> str:
    return hashlib.md5(x.astype(np.float32).tobytes()).hexdigest()


def load_label_mapping(
    metadata_path: Optional[Path],
    id_col: str,
    label_col: str,
) -> Dict[str, str]:
    """
    metadata CSV가 있으면 sample_id 또는 SEN ID -> label 텍스트 mapping 생성.
    없으면 빈 dict 반환.
    """
    if metadata_path is None:
        return {}

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    df = pd.read_csv(metadata_path, encoding="utf-8-sig")

    if id_col not in df.columns:
        raise ValueError(f"metadata에 id_col={id_col} 컬럼이 없습니다. columns={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"metadata에 label_col={label_col} 컬럼이 없습니다. columns={list(df.columns)}")

    mapping = {}

    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        label = str(row[label_col]).strip()

        if not raw_id or not label or label.lower() == "nan":
            continue

        mapping[raw_id] = label

        sen_id = parse_sen_id(raw_id)
        mapping[sen_id] = label

    return mapping


def get_label_for_sequence(seq_dir: Path, label_mapping: Dict[str, str]) -> str:
    sample_id = seq_dir.name
    sen_id = parse_sen_id(sample_id)

    if sample_id in label_mapping:
        return label_mapping[sample_id]

    if sen_id in label_mapping:
        return label_mapping[sen_id]

    # metadata가 없으면 SEN ID 자체를 라벨로 사용
    return sen_id


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def quality_check_X(X: np.ndarray) -> Dict[str, Any]:
    hashes = [sample_hash(X[i]) for i in range(len(X))]
    unique_hashes = set(hashes)

    report = {
        "shape": list(X.shape),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "zero_ratio": float(np.mean(X == 0)),
        "num_samples": int(len(X)),
        "num_unique_sample_hashes": int(len(unique_hashes)),
        "all_samples_identical": bool(len(unique_hashes) == 1),
    }

    return report


def split_dataset(X: np.ndarray, y: np.ndarray, val_ratio: float):
    """
    label별 샘플 수가 2개 이상이면 stratify 사용.
    1개짜리 label이 많으면 stratify 불가능하므로 일반 split 사용.
    """
    unique, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))

    stratify = y if min_count >= 2 and len(unique) > 1 else None

    return train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="AIHub F keypoint 폴더들이 들어 있는 상위 폴더",
    )

    parser.add_argument(
        "--metadata",
        default="",
        help="sample_id/SEN과 문장 라벨을 연결하는 CSV. 없으면 SEN ID를 라벨로 사용",
    )

    parser.add_argument(
        "--id-col",
        default="sample_id",
        help="metadata에서 sample id 컬럼명",
    )

    parser.add_argument(
        "--label-col",
        default="label",
        help="metadata에서 라벨 텍스트 컬럼명",
    )

    parser.add_argument(
        "--out",
        default="data",
        help="sen_AI 기준 output 폴더. 기본값 data",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="테스트용 최대 샘플 수. 0이면 전체",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    out_root = Path(args.out)

    metadata_path = Path(args.metadata) if args.metadata else None
    label_mapping = load_label_mapping(metadata_path, args.id_col, args.label_col)

    sequence_dirs = find_sequence_dirs(input_dir)

    if args.max_samples > 0:
        sequence_dirs = sequence_dirs[: args.max_samples]

    if not sequence_dirs:
        raise RuntimeError(f"F sequence folders not found under: {input_dir}")

    print("[INFO] input:", input_dir)
    print("[INFO] sequence dirs:", len(sequence_dirs))
    print("[INFO] metadata:", metadata_path)

    X_list = []
    labels = []
    meta_rows = []
    fail_rows = []

    for idx, seq_dir in enumerate(sequence_dirs, start=1):
        try:
            X_seq, raw_frame_count, nonzero_count = load_sequence(seq_dir)
            label = get_label_for_sequence(seq_dir, label_mapping)

            # 완전 빈 샘플 방지
            if nonzero_count == 0:
                fail_rows.append(
                    {
                        "sample_id": seq_dir.name,
                        "path": str(seq_dir),
                        "reason": "all_zero_keypoints",
                    }
                )
                continue

            X_list.append(X_seq)
            labels.append(label)

            meta_rows.append(
                {
                    "sample_id": seq_dir.name,
                    "sen_id": parse_sen_id(seq_dir.name),
                    "label": label,
                    "path": str(seq_dir),
                    "raw_frame_count": raw_frame_count,
                    "used_seq_len": SEQ_LEN,
                    "feature_dim": FEATURE_DIM,
                    "nonzero_count": nonzero_count,
                    "sample_hash": sample_hash(X_seq),
                }
            )

            if idx % 100 == 0:
                print(f"[INFO] processed {idx}/{len(sequence_dirs)}")

        except Exception as e:
            fail_rows.append(
                {
                    "sample_id": seq_dir.name,
                    "path": str(seq_dir),
                    "reason": str(e),
                }
            )

    if not X_list:
        raise RuntimeError("No valid sequences were processed.")

    X = np.stack(X_list, axis=0).astype(np.float32)

    classes = np.array(sorted(set(labels)), dtype=object)
    label_to_idx = {label: int(i) for i, label in enumerate(classes)}
    idx_to_label = {int(i): label for i, label in enumerate(classes)}
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)

    print("\n[INFO] built dataset")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("classes:", len(classes))

    dataset_quality = quality_check_X(X)

    if dataset_quality["all_samples_identical"]:
        raise RuntimeError(
            "전처리 결과 X의 모든 샘플이 동일합니다. "
            "keypoint 추출 로직 또는 입력 폴더를 확인하세요."
        )

    X_train, X_val, y_train, y_val = split_dataset(X, y, args.val_ratio)

    processed_dir = out_root / "processed"
    train_dir = out_root / "train"
    val_dir = out_root / "validation"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
    np.save(processed_dir / "classes.npy", classes)

    np.save(train_dir / "X_train.npy", X_train.astype(np.float32))
    np.save(train_dir / "y_train.npy", y_train.astype(np.int64))

    np.save(val_dir / "X_validation.npy", X_val.astype(np.float32))
    np.save(val_dir / "y_validation.npy", y_val.astype(np.int64))
    np.save(val_dir / "classes.npy", classes)

    label_map = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }

    save_json(processed_dir / "label_map.json", label_map)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(processed_dir / "preprocess_meta.csv", index=False, encoding="utf-8-sig")

    if fail_rows:
        fail_df = pd.DataFrame(fail_rows)
        fail_df.to_csv(processed_dir / "preprocess_failures.csv", index=False, encoding="utf-8-sig")

    train_quality = quality_check_X(X_train)
    val_quality = quality_check_X(X_val)

    report = {
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "num_total_sequences_found": len(sequence_dirs),
        "num_valid_sequences": int(len(X)),
        "num_failed_sequences": int(len(fail_rows)),
        "num_classes": int(len(classes)),
        "classes_preview": classes[:20].tolist(),
        "dataset_quality": dataset_quality,
        "train_quality": train_quality,
        "validation_quality": val_quality,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "label_counts": {
            str(classes[int(label_idx)]): int(count)
            for label_idx, count in zip(*np.unique(y, return_counts=True))
        },
    }

    save_json(processed_dir / "preprocess_report.json", report)

    print("\n[DONE] saved files")
    print("processed:", processed_dir)
    print("train:", train_dir)
    print("validation:", val_dir)

    print("\n[QUALITY]")
    print(json.dumps(report["dataset_quality"], ensure_ascii=False, indent=2))
    print(json.dumps(report["validation_quality"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()