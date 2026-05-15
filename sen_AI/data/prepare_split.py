from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
VAL_SIZE = 0.20
INPUT_SHAPE = (30, 120)
N_SPLITS = int(1 / VAL_SIZE)
METADATA_FILENAME = "used_videos_final_CLEANED.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"


def resolve_raw_dir():
    candidates = [
        RAW_DIR,
        PROJECT_ROOT / "전처리 완료(수정 완)",
        PROJECT_ROOT,
    ]

    for path in candidates:
        if (path / "Batch_001_X.npy").exists() and (path / "classes.npy").exists():
            return path

    for path in PROJECT_ROOT.iterdir():
        if path.is_dir() and (path / "Batch_001_X.npy").exists() and (path / "classes.npy").exists():
            return path

    raise FileNotFoundError(
        "Raw batch data not found. Put Batch_###_X/y.npy and classes.npy in data/raw."
    )


def resolve_metadata_path(raw_dir):
    candidates = [
        raw_dir / METADATA_FILENAME,
        PROJECT_ROOT / METADATA_FILENAME,
        PROJECT_ROOT / "전처리 완료(수정 완)" / METADATA_FILENAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(PROJECT_ROOT.rglob(METADATA_FILENAME))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Metadata CSV not found: {METADATA_FILENAME}")


def load_batches(base):
    x_list = []
    y_list = []

    for i in range(1, 34):
        x_path = base / f"Batch_{i:03d}_X.npy"
        y_path = base / f"Batch_{i:03d}_y.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"Missing X file: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing y file: {y_path}")

        x_batch = np.load(x_path).astype(np.float32)
        y_batch = np.load(y_path).astype(np.int64)

        print(f"Batch {i:03d}: X {x_batch.shape}, y {y_batch.shape}")
        x_list.append(x_batch)
        y_list.append(y_batch)

    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    classes = np.load(base / "classes.npy", allow_pickle=True)

    assert X.shape[1:] == INPUT_SHAPE
    assert y.ndim == 1
    assert y.min() >= 0
    assert y.max() < len(classes)

    print("\nLoaded data")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("classes:", len(classes))

    return X, y, classes


def load_groups(metadata_path, y):
    meta_df = pd.read_csv(metadata_path)

    required_columns = {"video_uid", "label_idx"}
    missing_columns = required_columns - set(meta_df.columns)
    if missing_columns:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing_columns)}")

    if len(meta_df) != len(y):
        raise ValueError(
            f"Metadata row count does not match y length: {len(meta_df)} != {len(y)}"
        )

    label_idx = meta_df["label_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(label_idx, y):
        raise ValueError("Metadata label_idx does not match concatenated y labels.")

    groups = meta_df["video_uid"].astype(str).str.extract(r"(SEN\d+)", expand=False)
    if groups.isna().any():
        bad_count = int(groups.isna().sum())
        raise ValueError(f"Could not extract SEN group from {bad_count} video_uid values.")

    groups = groups.to_numpy()

    print("\nLoaded metadata")
    print("metadata:", metadata_path)
    print("rows:", len(meta_df))
    print("unique SEN groups:", len(np.unique(groups)))

    return groups


def split_by_sen_group(X, y, groups):
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    train_idx, validation_idx = next(splitter.split(X, y, groups))

    train_groups = set(groups[train_idx])
    validation_groups = set(groups[validation_idx])
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError(f"SEN group leakage detected: {len(overlap)} overlapping groups")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_validation = X[validation_idx]
    y_validation = y[validation_idx]

    missing_in_train = sorted(set(np.unique(y)) - set(np.unique(y_train)))
    missing_in_validation = sorted(set(np.unique(y)) - set(np.unique(y_validation)))

    print("\nSEN group split check")
    print("train samples:", len(train_idx))
    print("validation samples:", len(validation_idx))
    print("train SEN groups:", len(train_groups))
    print("validation SEN groups:", len(validation_groups))
    print("overlapping SEN groups:", len(overlap))
    print("train unique labels:", len(np.unique(y_train)))
    print("validation unique labels:", len(np.unique(y_validation)))
    print("labels missing in train:", len(missing_in_train))
    print("labels missing in validation:", len(missing_in_validation))

    return X_train, X_validation, y_train, y_validation


def main():
    base = resolve_raw_dir()
    print("raw data folder:", base)

    X, y, classes = load_batches(base)
    metadata_path = resolve_metadata_path(base)
    groups = load_groups(metadata_path, y)

    X_train, X_validation, y_train, y_validation = split_by_sen_group(X, y, groups)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    np.save(TRAIN_DIR / "X_train.npy", X_train.astype(np.float32))
    np.save(TRAIN_DIR / "y_train.npy", y_train.astype(np.int64))
    np.save(VALIDATION_DIR / "X_validation.npy", X_validation.astype(np.float32))
    np.save(VALIDATION_DIR / "y_validation.npy", y_validation.astype(np.int64))
    np.save(VALIDATION_DIR / "classes.npy", classes)

    print("\nSaved split")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)
    print("train folder:", TRAIN_DIR)
    print("validation folder:", VALIDATION_DIR)


if __name__ == "__main__":
    main()
