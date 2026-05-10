import argparse
import json
import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from degree_features import normalize_face_points, extract_degree_features_from_points


AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
ANGER_AUS = [4, 17, 20, 25, 26]
LABEL_NAMES = ["weak", "normal", "strong"]


def safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = Path(member)

            if member.startswith("/") or ".." in member_path.parts:
                continue

            zf.extract(member, out_dir)


def prepare_data(raw_dir: Path, work_dir: Path) -> tuple[Path, Path]:
    action_zip = raw_dir / "ActionUnit_Labels.zip"
    landmark_zip = raw_dir / "Landmark_Points.zip"

    if not action_zip.exists():
        raise FileNotFoundError(f"Missing file: {action_zip}")

    if not landmark_zip.exists():
        raise FileNotFoundError(f"Missing file: {landmark_zip}")

    extracted_dir = work_dir / "extracted"
    action_out = extracted_dir / "ActionUnit_Labels"
    landmark_out = extracted_dir / "Landmark_Points"

    if not action_out.exists():
        print("[1/3] Extracting ActionUnit_Labels.zip ...")
        safe_extract_zip(action_zip, action_out)

    if not landmark_out.exists():
        print("[2/3] Extracting Landmark_Points.zip ...")
        safe_extract_zip(landmark_zip, extracted_dir)

    nested_zips = sorted(landmark_out.rglob("SN*.zip"))
    if nested_zips:
        print("[3/3] Extracting nested landmark subject zips ...")
        for z in tqdm(nested_zips):
            subject_name = z.stem
            subject_out = z.parent / subject_name

            if subject_out.exists():
                continue

            subject_out.mkdir(parents=True, exist_ok=True)
            safe_extract_zip(z, subject_out)

    return action_out, landmark_out


def _to_int(value):
    if pd.isna(value):
        return None

    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "":
                return None
            return int(float(value))
        return int(float(value))
    except Exception:
        return None


def load_disfa_error_ranges(raw_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """
    Error_LOG_Sheet.xls에서 landmark tracking 오류 frame range를 읽는다.
    """
    xls_path = raw_dir / "Error_LOG_Sheet.xls"

    if not xls_path.exists():
        print(f"[WARN] Error_LOG_Sheet not found: {xls_path}")
        return {}

    df = pd.read_excel(xls_path, header=None)

    error_ranges: dict[str, list[tuple[int, int]]] = {}

    for _, row in df.iloc[5:].iterrows():
        subject = row.iloc[0]

        if pd.isna(subject):
            continue

        subject = str(subject).strip()

        if not re.fullmatch(r"SN\d+", subject):
            continue

        error_ranges.setdefault(subject, [])

        for col in range(1, len(row), 2):
            if col + 1 >= len(row):
                break

            start = _to_int(row.iloc[col])
            end = _to_int(row.iloc[col + 1])

            if start is None or end is None:
                continue

            if start > end:
                start, end = end, start

            error_ranges[subject].append((start, end))

    total_ranges = sum(len(v) for v in error_ranges.values())
    total_frames = sum(
        end - start + 1
        for ranges in error_ranges.values()
        for start, end in ranges
    )

    print(
        f"[INFO] Loaded DISFA error ranges: "
        f"{len(error_ranges)} subjects, {total_ranges} ranges, {total_frames} frames"
    )

    return error_ranges


def is_error_frame(
    error_ranges: dict[str, list[tuple[int, int]]],
    subject: str,
    frame: int,
) -> bool:
    for start, end in error_ranges.get(subject, []):
        if start <= frame <= end:
            return True
    return False


def read_au_file(path: Path) -> pd.DataFrame:
    m = re.search(r"_au(\d+)\.txt$", path.name.lower())

    if not m:
        raise ValueError(f"Cannot parse AU number from file name: {path.name}")

    au_num = int(m.group(1))
    col = f"AU{au_num}"

    df = pd.read_csv(
        path,
        header=None,
        sep=r"[\s,]+",
        engine="python",
    )

    if df.shape[1] < 2:
        raise ValueError(f"Bad AU file format: {path}")

    df = df.iloc[:, :2]
    df.columns = ["frame", col]
    df["frame"] = df["frame"].astype(int)
    df[col] = df[col].astype(float)

    return df


def find_au_files(action_dir: Path, subject: str) -> list[Path]:
    files = sorted(action_dir.rglob(f"{subject}_au*.txt"))

    if not files:
        files = sorted(action_dir.rglob(f"*{subject}*au*.txt"))

    return files


def load_subject_au_labels(action_dir: Path, subject: str) -> pd.DataFrame:
    files = find_au_files(action_dir, subject)

    if not files:
        raise FileNotFoundError(f"No AU label files found for {subject}")

    merged = None

    for f in files:
        df = read_au_file(f)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="frame", how="outer")

    if merged is None:
        raise RuntimeError(f"Failed to load AU labels for {subject}")

    merged = merged.sort_values("frame").reset_index(drop=True)

    for au in AU_LIST:
        col = f"AU{au}"
        if col not in merged.columns:
            merged[col] = 0.0

    merged = merged[["frame"] + [f"AU{au}" for au in AU_LIST]]
    merged = merged.fillna(0.0)

    return merged


def parse_frame_from_landmark_file(path: Path) -> int | None:
    m = re.search(r"_(\d+)_lm\.mat$", path.name.lower())

    if not m:
        return None

    return int(m.group(1))


def find_subject_landmark_dirs(landmark_root: Path) -> dict[str, Path]:
    subject_dirs = {}

    for p in landmark_root.rglob("*"):
        if p.is_dir() and re.fullmatch(r"SN\d+", p.name):
            mat_files = list(p.rglob("*_lm.mat"))

            if mat_files:
                subject_dirs[p.name] = p

    return dict(sorted(subject_dirs.items()))


def load_landmark_points(path: Path) -> np.ndarray:
    mat = sio.loadmat(path)

    if "pts" not in mat:
        raise KeyError(f"'pts' key not found in {path}")

    pts = np.asarray(mat["pts"], dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Bad landmark shape {pts.shape} in {path}")

    return pts


def make_raw_score(row: pd.Series, mode: str) -> float:
    if mode == "anger":
        values = [float(row[f"AU{au}"]) for au in ANGER_AUS if f"AU{au}" in row]
        return float(max(values)) if values else 0.0

    if mode == "overall":
        values = [float(row[f"AU{au}"]) for au in AU_LIST if f"AU{au}" in row]
        return float(max(values)) if values else 0.0

    raise ValueError("mode must be 'anger' or 'overall'")


def score_to_label(score: float) -> int:
    """
    DISFA intensity 0~5를 3단계로 변환.
    0~1: weak
    2~3: normal
    4~5: strong
    """
    if score <= 1:
        return 0
    if score <= 3:
        return 1
    return 2


def make_280_frame_feature(
    points: np.ndarray,
    prev_norm_flat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    280차원 frame feature 생성.

    16차원 요약 feature
    + 132차원 정규화 landmark
    + 132차원 delta
    = 280차원
    """
    summary_16 = extract_degree_features_from_points(points)

    norm_points = normalize_face_points(points)
    norm_flat = norm_points.flatten().astype(np.float32)

    if norm_flat.shape[0] != 132:
        raise ValueError(f"Expected 132 landmark dims, got {norm_flat.shape[0]}")

    if prev_norm_flat is None:
        delta = np.zeros_like(norm_flat, dtype=np.float32)
    else:
        delta = norm_flat - prev_norm_flat

    feature = np.concatenate([summary_16, norm_flat, delta], axis=0).astype(np.float32)

    if feature.shape[0] != 280:
        raise ValueError(f"Expected 280 feature dims, got {feature.shape[0]}")

    return feature, norm_flat


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_wide_and_sequence_dataset(
    raw_dir: Path,
    work_dir: Path,
    out_dir: Path,
    mode: str,
    seq_len: int,
    stride: int,
) -> None:
    action_dir, landmark_root = prepare_data(raw_dir, work_dir)
    error_ranges = load_disfa_error_ranges(raw_dir)

    subject_dirs = find_subject_landmark_dirs(landmark_root)

    if not subject_dirs:
        raise RuntimeError(f"No subject landmark folders found under {landmark_root}")

    print(f"Found {len(subject_dirs)} subject folders.")
    print(list(subject_dirs.keys()))

    X_wide = []
    y_wide = []
    meta_wide_rows = []

    X_seq = []
    y_seq = []
    meta_seq_rows = []

    error_rows = []

    removed_by_error_log = 0
    skipped_no_au = 0

    for subject, subject_dir in subject_dirs.items():
        print(f"\nProcessing subject: {subject}")

        try:
            au_df = load_subject_au_labels(action_dir, subject)
        except Exception as e:
            print(f"[WARN] AU load failed for {subject}: {e}")
            error_rows.append({
                "subject": subject,
                "frame": None,
                "file": None,
                "error": f"AU load failed: {e}",
            })
            continue

        au_by_frame = {
            int(row["frame"]): row
            for _, row in au_df.iterrows()
        }

        lm_files = sorted(
            subject_dir.rglob("*_lm.mat"),
            key=lambda p: parse_frame_from_landmark_file(p) or -1,
        )

        subject_features = []
        subject_scores = []
        subject_labels = []
        subject_frames = []

        prev_norm_flat = None

        for lm_path in tqdm(lm_files, desc=subject):
            frame = parse_frame_from_landmark_file(lm_path)

            if frame is None:
                continue

            if is_error_frame(error_ranges, subject, frame):
                removed_by_error_log += 1
                continue

            if frame not in au_by_frame:
                skipped_no_au += 1
                continue

            try:
                pts = load_landmark_points(lm_path)
                feature_280, prev_norm_flat = make_280_frame_feature(pts, prev_norm_flat)

                score = make_raw_score(au_by_frame[frame], mode=mode)
                label = score_to_label(score)

                X_wide.append(feature_280)
                y_wide.append(label)

                meta_wide_rows.append({
                    "subject": subject,
                    "frame": frame,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "score": score,
                    "feature_dim": 280,
                    "landmark_file": str(lm_path),
                    **{f"AU{au}": float(au_by_frame[frame][f"AU{au}"]) for au in AU_LIST},
                })

                subject_features.append(feature_280)
                subject_scores.append(score)
                subject_labels.append(label)
                subject_frames.append(frame)

            except Exception as e:
                error_rows.append({
                    "subject": subject,
                    "frame": frame,
                    "file": str(lm_path),
                    "error": str(e),
                })

        if len(subject_features) < seq_len:
            print(f"[WARN] {subject}: not enough valid frames for sequence ({len(subject_features)})")
            continue

        subject_features = np.asarray(subject_features, dtype=np.float32)
        subject_scores = np.asarray(subject_scores, dtype=np.float32)
        subject_frames = np.asarray(subject_frames, dtype=np.int64)

        for start in range(0, len(subject_features) - seq_len + 1, stride):
            end = start + seq_len

            seq_x = subject_features[start:end]

            seq_score = float(np.max(subject_scores[start:end]))
            seq_label = score_to_label(seq_score)

            X_seq.append(seq_x)
            y_seq.append(seq_label)

            meta_seq_rows.append({
                "subject": subject,
                "start_frame": int(subject_frames[start]),
                "end_frame": int(subject_frames[end - 1]),
                "label": int(seq_label),
                "label_name": LABEL_NAMES[seq_label],
                "score": seq_score,
                "seq_len": seq_len,
                "stride": stride,
                "feature_dim": int(seq_x.shape[1]),
            })

    if not X_wide:
        raise RuntimeError("No wide frame samples were created.")

    if not X_seq:
        raise RuntimeError("No sequence samples were created.")

    X_wide = np.asarray(X_wide, dtype=np.float32)
    y_wide = np.asarray(y_wide, dtype=np.int64)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-frame 280D output
    np.save(out_dir / f"Xwide_{mode}.npy", X_wide)
    np.save(out_dir / f"ywide_{mode}.npy", y_wide)

    meta_wide_df = pd.DataFrame(meta_wide_rows)
    meta_wide_df.to_csv(
        out_dir / f"metawide_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    label_map_wide = {
        "label_to_idx": {
            "weak": 0,
            "normal": 1,
            "strong": 2,
        },
        "idx_to_label": {
            "0": "weak",
            "1": "normal",
            "2": "strong",
        },
        "mode": mode,
        "feature_type": "summary16 + normalized_landmark132 + delta132",
        "feature_dim": 280,
        "input_shape": [280],
        "source": "DISFA Landmark_Points + ActionUnit_Labels",
        "error_log_sheet_used": True,
    }

    preprocess_meta_wide = {
        "mode": mode,
        "dataset_type": "1_frame_280d",
        "num_samples": int(len(X_wide)),
        "feature_dim": 280,
        "x_shape": list(X_wide.shape),
        "y_shape": list(y_wide.shape),
        "label_rule": "0~1 weak, 2~3 normal, 4~5 strong",
        "feature_rule": "16 summary features + 132 normalized facial landmarks + 132 temporal delta",
        "anger_aus": ANGER_AUS,
        "au_list": AU_LIST,
        "error_log_sheet_used": True,
        "removed_by_error_log": int(removed_by_error_log),
        "skipped_no_au": int(skipped_no_au),
    }

    save_json(out_dir / f"label_map_wide_{mode}.json", label_map_wide)
    save_json(out_dir / f"preprocess_meta_wide_{mode}.json", preprocess_meta_wide)
    joblib.dump(label_map_wide, out_dir / f"label_info_wide_{mode}.joblib")

    # 30-frame sequence output
    np.save(out_dir / f"Xseq_{mode}.npy", X_seq)
    np.save(out_dir / f"yseq_{mode}.npy", y_seq)

    meta_seq_df = pd.DataFrame(meta_seq_rows)
    meta_seq_df.to_csv(
        out_dir / f"metaseq_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    label_map_seq = {
        "label_to_idx": {
            "weak": 0,
            "normal": 1,
            "strong": 2,
        },
        "idx_to_label": {
            "0": "weak",
            "1": "normal",
            "2": "strong",
        },
        "mode": mode,
        "seq_len": seq_len,
        "stride": stride,
        "feature_type": "summary16 + normalized_landmark132 + delta132",
        "feature_dim": 280,
        "input_shape": [seq_len, 280],
        "source": "DISFA Landmark_Points + ActionUnit_Labels",
        "error_log_sheet_used": True,
    }

    preprocess_meta_seq = {
        "mode": mode,
        "dataset_type": "30_frame_280d_sequence",
        "num_sequences": int(len(X_seq)),
        "seq_len": seq_len,
        "stride": stride,
        "feature_dim": 280,
        "x_shape": list(X_seq.shape),
        "y_shape": list(y_seq.shape),
        "label_rule": "sequence score=max frame score; 0~1 weak, 2~3 normal, 4~5 strong",
        "feature_rule": "16 summary features + 132 normalized facial landmarks + 132 temporal delta",
        "anger_aus": ANGER_AUS,
        "au_list": AU_LIST,
        "error_log_sheet_used": True,
        "removed_by_error_log": int(removed_by_error_log),
        "skipped_no_au": int(skipped_no_au),
    }

    save_json(out_dir / f"label_map_seq_{mode}.json", label_map_seq)
    save_json(out_dir / f"preprocess_meta_seq_{mode}.json", preprocess_meta_seq)
    joblib.dump(label_map_seq, out_dir / f"label_info_seq_{mode}.joblib")

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(
        out_dir / f"error_log_wide_seq_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[DONE] DISFA 280D frame and 30x280D sequence datasets created.")

    print("\n[1-frame 280D]")
    print(f"Xwide shape: {X_wide.shape}")
    print(f"ywide shape: {y_wide.shape}")
    print("Label counts:")
    print(meta_wide_df["label_name"].value_counts())

    print("\n[30-frame x 280D sequence]")
    print(f"Xseq shape: {X_seq.shape}")
    print(f"yseq shape: {y_seq.shape}")
    print("Label counts:")
    print(meta_seq_df["label_name"].value_counts())

    print(f"\nRemoved by Error_LOG_Sheet: {removed_by_error_log}")
    print(f"Skipped no AU frame: {skipped_no_au}")
    print(f"Saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--work_dir", type=str, default="data/work")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--mode", type=str, default="anger", choices=["anger", "overall"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    build_wide_and_sequence_dataset(
        raw_dir=Path(args.raw_dir),
        work_dir=Path(args.work_dir),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        seq_len=args.seq_len,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()