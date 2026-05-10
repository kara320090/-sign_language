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

from degree_features import extract_degree_features_from_points
from error_log_utils import load_disfa_error_ranges, is_error_frame


AU_LIST = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

# DISFA에는 AU7, AU23, AU24가 없음.
# 화남/강한 표현 쪽은 AU4 중심으로 구성.
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

    # Landmark_Points 안에 SN001.zip 같은 중첩 zip이 있으면 한 번 더 해제
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
    """
    AU txt 파일을 읽는다.
    예상 파일명:
    SN001_au1.txt, SN001_au4.txt 등
    예상 내용:
    frame, intensity
    """
    m = re.search(r"_au(\d+)\.txt$", path.name.lower())
    if not m:
        raise ValueError(f"Cannot parse AU number from file name: {path.name}")

    au_num = int(m.group(1))
    col = f"AU{au_num}"

    # 콤마/공백/탭 모두 대응
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
    """
    예상 파일명:
    SN001_0001_lm.mat
    SN001_1_lm.mat
    """
    m = re.search(r"_(\d+)_lm\.mat$", path.name.lower())
    if not m:
        return None
    return int(m.group(1))


def find_subject_landmark_dirs(landmark_root: Path) -> dict[str, Path]:
    """
    SN001, SN002 ... subject 폴더를 찾는다.
    """
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
        # 화난 정도 후보: AU4 중심 + 입/턱 움직임 보조
        values = [float(row[f"AU{au}"]) for au in ANGER_AUS if f"AU{au}" in row]
        if not values:
            return 0.0
        return float(max(values))

    if mode == "overall":
        values = [float(row[f"AU{au}"]) for au in AU_LIST if f"AU{au}" in row]
        if not values:
            return 0.0
        return float(max(values))

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


def build_dataset(raw_dir: Path, work_dir: Path, out_dir: Path, mode: str) -> None:
    action_dir, landmark_root = prepare_data(raw_dir, work_dir)

    # Error_LOG_Sheet 기반 이상 프레임 구간 로드
    error_ranges = load_disfa_error_ranges(raw_dir)

    subject_dirs = find_subject_landmark_dirs(landmark_root)

    if not subject_dirs:
        raise RuntimeError(f"No subject landmark folders found under {landmark_root}")

    print(f"Found {len(subject_dirs)} subject folders.")
    print(list(subject_dirs.keys()))

    X = []
    y = []
    rows = []
    errors = []

    removed_by_error_log = 0
    skipped_no_au = 0

    for subject, subject_dir in subject_dirs.items():
        print(f"\nProcessing subject: {subject}")

        try:
            au_df = load_subject_au_labels(action_dir, subject)
        except Exception as e:
            print(f"[WARN] AU load failed for {subject}: {e}")
            errors.append({
                "subject": subject,
                "frame": None,
                "file": None,
                "error": f"AU load failed: {e}"
            })
            continue

        au_by_frame = {
            int(row["frame"]): row
            for _, row in au_df.iterrows()
        }

        lm_files = sorted(subject_dir.rglob("*_lm.mat"))

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
                feat = extract_degree_features_from_points(pts)

                score = make_raw_score(au_by_frame[frame], mode=mode)
                label = score_to_label(score)

                X.append(feat)
                y.append(label)

                rows.append({
                    "subject": subject,
                    "frame": frame,
                    "label": label,
                    "label_name": ["weak", "normal", "strong"][label],
                    "score": score,
                    "landmark_file": str(lm_path),
                    **{f"AU{au}": float(au_by_frame[frame][f"AU{au}"]) for au in AU_LIST}
                })

            except Exception as e:
                errors.append({
                    "subject": subject,
                    "frame": frame,
                    "file": str(lm_path),
                    "error": str(e)
                })

    if not X:
        raise RuntimeError("No training samples were created.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"X_{mode}.npy", X)
    np.save(out_dir / f"y_{mode}.npy", y)

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(out_dir / f"meta_{mode}.csv", index=False, encoding="utf-8-sig")

    error_df = pd.DataFrame(errors)
    error_df.to_csv(out_dir / f"error_log_{mode}.csv", index=False, encoding="utf-8-sig")

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
        "feature_type": "common_face_degree_features",
        "feature_dim": int(X.shape[1]),
        "source": "DISFA Landmark_Points + ActionUnit_Labels",
        "error_log_sheet_used": True
    }

    with open(out_dir / f"label_map_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    preprocess_meta = {
        "mode": mode,
        "num_samples": int(len(X)),
        "feature_dim": int(X.shape[1]),
        "label_rule": "0~1 weak, 2~3 normal, 4~5 strong",
        "anger_aus": ANGER_AUS,
        "au_list": AU_LIST,
        "error_log_sheet_used": True,
        "removed_by_error_log": int(removed_by_error_log),
        "skipped_no_au": int(skipped_no_au),
        "note": "This dataset is for degree_AI only. It is separate from word/sen AIHub preprocessing."
    }

    with open(out_dir / f"preprocess_meta_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_meta, f, ensure_ascii=False, indent=2)

    joblib.dump(label_map, out_dir / f"label_info_{mode}.joblib")

    print("\n[DONE] DISFA degree dataset created.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
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
    args = parser.parse_args()

    build_dataset(
        raw_dir=Path(args.raw_dir),
        work_dir=Path(args.work_dir),
        out_dir=Path(args.out_dir),
        mode=args.mode
    )


if __name__ == "__main__":
    main()