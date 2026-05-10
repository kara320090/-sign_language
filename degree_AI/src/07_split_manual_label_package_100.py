import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]


def find_video(video_root: Path, sequence_name: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = video_root / f"{sequence_name}{ext}"
        if p.exists():
            return p

    matches = list(video_root.rglob(f"{sequence_name}.*"))
    if matches:
        return matches[0]

    return None


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_sample_reason(row) -> str:
    pw = float(row.get("prob_weak", 0.0))
    pn = float(row.get("prob_normal", 0.0))
    ps = float(row.get("prob_strong", 0.0))

    if ps >= pw and ps >= pn:
        return "strong_candidate"
    if pw >= pn and pw >= ps:
        return "weak_candidate"
    return "normal_candidate"


def make_package(
    rows: list[dict],
    video_root: Path,
    package_dir: Path,
    assignee: str,
) -> None:
    video_out = package_dir / "videos"
    video_out.mkdir(parents=True, exist_ok=True)

    package_rows = []

    for idx, row in enumerate(rows):
        sequence_name = row["sequence_name"]
        video_path = find_video(video_root, sequence_name)

        copied_video_name = ""
        copied_video_path = ""

        if video_path is not None:
            dst = video_out / video_path.name
            shutil.copy2(video_path, dst)
            copied_video_name = dst.name
            copied_video_path = str(dst)

        package_rows.append({
            "sample_id": f"{assignee}_{idx + 1:03d}",
            "assignee": assignee,
            "sequence_name": sequence_name,
            "video_file": copied_video_name,
            "video_path": copied_video_path,
            "keypoint_dir": row.get("sequence_dir", ""),
            "pred_degree": row.get("degree", ""),
            "confidence": float(row.get("confidence", 0.0)),
            "prob_weak": float(row.get("prob_weak", 0.0)),
            "prob_normal": float(row.get("prob_normal", 0.0)),
            "prob_strong": float(row.get("prob_strong", 0.0)),
            "sample_reason": row.get("sample_reason", ""),
            "manual_degree": "",
            "memo": ""
        })

    csv_path = package_dir / f"manual_degree_label_{assignee}_50.csv"
    json_path = package_dir / f"manual_degree_label_{assignee}_50.json"

    pd.DataFrame(package_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_json(json_path, package_rows)

    print(f"[DONE] {assignee} package")
    print(f"CSV   : {csv_path}")
    print(f"JSON  : {json_path}")
    print(f"Videos: {video_out}")
    print(f"Count : {len(package_rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--out_dir", default="outputs/manual_label_split_100")
    parser.add_argument("--my_name", default="bongheon")
    parser.add_argument("--teammate_name", default="teammate")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    df = df[df["status"] == "ok"].copy()

    for col in ["prob_weak", "prob_normal", "prob_strong", "confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 100개 전체를 사용하되, 강한 후보/약한 후보가 한쪽에 몰리지 않도록 정렬 후 교차 분배
    df["sample_reason"] = df.apply(make_sample_reason, axis=1)

    df = df.sort_values(
        by=["prob_strong", "prob_weak", "prob_normal", "confidence"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    rows = df.to_dict(orient="records")

    my_rows = rows[0::2][:50]
    teammate_rows = rows[1::2][:50]

    # 혹시 홀짝 분배로 50개가 안 맞을 경우 보정
    if len(my_rows) < 50:
        used = {r["sequence_name"] for r in my_rows + teammate_rows}
        extra = [r for r in rows if r["sequence_name"] not in used]
        my_rows.extend(extra[:50 - len(my_rows)])

    if len(teammate_rows) < 50:
        used = {r["sequence_name"] for r in my_rows + teammate_rows}
        extra = [r for r in rows if r["sequence_name"] not in used]
        teammate_rows.extend(extra[:50 - len(teammate_rows)])

    make_package(
        rows=my_rows,
        video_root=video_root,
        package_dir=out_dir / f"{args.my_name}_50",
        assignee=args.my_name,
    )

    make_package(
        rows=teammate_rows,
        video_root=video_root,
        package_dir=out_dir / f"{args.teammate_name}_50",
        assignee=args.teammate_name,
    )

    combined_rows = []
    combined_rows.extend(my_rows)
    combined_rows.extend(teammate_rows)

    combined_df = pd.DataFrame(combined_rows)
    combined_path = out_dir / "manual_degree_label_split_summary.csv"
    combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    print(f"Total ok samples: {len(rows)}")
    print(f"{args.my_name}: {len(my_rows)}")
    print(f"{args.teammate_name}: {len(teammate_rows)}")
    print(f"Combined summary: {combined_path}")

    print("\nSample reason counts:")
    print(pd.DataFrame(combined_rows)["sample_reason"].value_counts())


if __name__ == "__main__":
    main()