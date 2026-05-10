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
from error_log_utils import load_disfa_error_ranges, is_error_frame


AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
ANGER_AUS = [4, 17, 20, 25, 26]


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
        engine="python"
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
    if score <= 1:
        return 0  # weak
    if score <= 3:
        return 1  # normal
    return 2      # strong


def make_wide_frame_feature(
    points: np.ndarray,
    prev_norm_flat: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    반환:
    feature = 16 summary + 132 normalized landmark + 132 delta = 280차원
    norm_flat = 다음 프레임 delta 계산용
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

    return feature, norm_flat


def build_sequence_dataset(
    raw_dir: Path,
    work_dir: Path,
    out_dir: Path,
    mode: str,
    seq_len: int,
    stride: int,
) -> None:
    action_dir, landmark_root = prepare_data(raw_dir, work_dir)

    # Error_LOG_Sheet 기반 이상 프레임 구간 로드
    error_ranges = load_disfa_error_ranges(raw_dir)

    subject_dirs = find_subject_landmark_dirs(landmark_root)

    if not subject_dirs:
        raise RuntimeError(f"No subject landmark folders found under {landmark_root}")

    print(f"Found {len(subject_dirs)} subject folders.")
    print(list(subject_dirs.keys()))

    X_seq = []
    y_seq = []
    meta_rows = []
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
                "error": f"AU load failed: {e}"
            })
            continue

        au_by_frame = {int(row["frame"]): row for _, row in au_df.iterrows()}

        lm_files = sorted(
            subject_dir.rglob("*_lm.mat"),
            key=lambda p: parse_frame_from_landmark_file(p) or -1
        )

        frame_features = []
        frame_scores = []
        frame_numbers = []

        prev_norm_flat = None

        for lm_path in tqdm(lm_files, desc=subject):
            frame = parse_frame_from_landmark_file(lm_path)

            if frame is None:
                continue

            # Error_LOG_Sheet에 기록된 오류 프레임 제거
            if is_error_frame(error_ranges, subject, frame):
                removed_by_error_log += 1
                continue

            if frame not in au_by_frame:
                skipped_no_au += 1
                continue

            try:
                pts = load_landmark_points(lm_path)
                feature, prev_norm_flat = make_wide_frame_feature(pts, prev_norm_flat)

                score = make_raw_score(au_by_frame[frame], mode=mode)

                frame_features.append(feature)
                frame_scores.append(score)
                frame_numbers.append(frame)

            except Exception as e:
                error_rows.append({
                    "subject": subject,
                    "frame": frame,
                    "file": str(lm_path),
                    "error": str(e)
                })

        if len(frame_features) < seq_len:
            print(f"[WARN] {subject}: not enough frames ({len(frame_features)})")
            continue

        frame_features = np.asarray(frame_features, dtype=np.float32)
        frame_scores = np.asarray(frame_scores, dtype=np.float32)
        frame_numbers = np.asarray(frame_numbers, dtype=np.int64)

        for start in range(0, len(frame_features) - seq_len + 1, stride):
            end = start + seq_len

            seq_x = frame_features[start:end]

            # 시퀀스 안에서 가장 강한 표정을 기준으로 라벨 결정
            seq_score = float(np.max(frame_scores[start:end]))
            seq_label = score_to_label(seq_score)

            X_seq.append(seq_x)
            y_seq.append(seq_label)

            meta_rows.append({
                "subject": subject,
                "start_frame": int(frame_numbers[start]),
                "end_frame": int(frame_numbers[end - 1]),
                "label": int(seq_label),
                "label_name": ["weak", "normal", "strong"][seq_label],
                "score": seq_score,
                "seq_len": seq_len,
                "stride": stride,
                "feature_dim": int(seq_x.shape[1])
            })

    if not X_seq:
        raise RuntimeError("No sequence samples were created.")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"Xseq_{mode}.npy", X_seq)
    np.save(out_dir / f"yseq_{mode}.npy", y_seq)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / f"metaseq_{mode}.csv", index=False, encoding="utf-8-sig")

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(out_dir / f"error_log_seq_{mode}.csv", index=False, encoding="utf-8-sig")

    label_map = {
        "label_to_idx": {
            "weak": 0,
            "normal": 1,
            "strong": 2
        },
        "idx_to_label": {
            "0": "weak",
            "1": "normal",
            "2": "strong"
        },
        "mode": mode,
        "seq_len": seq_len,
        "stride": stride,
        "feature_type": "summary16 + normalized_landmark132 + delta132",
        "feature_dim": int(X_seq.shape[2]),
        "source": "DISFA Landmark_Points + ActionUnit_Labels",
        "error_log_sheet_used": True
    }

    with open(out_dir / f"label_map_seq_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    preprocess_meta = {
        "mode": mode,
        "num_sequences": int(len(X_seq)),
        "seq_len": seq_len,
        "stride": stride,
        "feature_dim": int(X_seq.shape[2]),
        "x_shape": list(X_seq.shape),
        "y_shape": list(y_seq.shape),
        "label_rule": "sequence score=max frame score; 0~1 weak, 2~3 normal, 4~5 strong",
        "feature_rule": "16 summary features + 132 normalized facial landmarks + 132 temporal delta",
        "anger_aus": ANGER_AUS,
        "au_list": AU_LIST,
        "error_log_sheet_used": True,
        "removed_by_error_log": int(removed_by_error_log),
        "skipped_no_au": int(skipped_no_au)
    }

    with open(out_dir / f"preprocess_meta_seq_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_meta, f, ensure_ascii=False, indent=2)

    joblib.dump(label_map, out_dir / f"label_info_seq_{mode}.joblib")

    print("\n[DONE] DISFA sequence degree dataset created.")
    print(f"Xseq shape: {X_seq.shape}")
    print(f"yseq shape: {y_seq.shape}")
    print(f"Removed by Error_LOG_Sheet: {removed_by_error_log}")
    print(f"Skipped no AU frame: {skipped_no_au}")
    print("\nLabel counts:")
    print(meta_df["label_name"].value_counts())
    print(f"\nSaved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--work_dir", type=str, default="data/work")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--mode", type=str, default="anger", choices=["anger", "overall"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    build_sequence_dataset(
        raw_dir=Path(args.raw_dir),
        work_dir=Path(args.work_dir),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        seq_len=args.seq_len,
        stride=args.stride
    )


if __name__ == "__main__":
    main()