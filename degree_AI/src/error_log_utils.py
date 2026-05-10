from pathlib import Path
import re
import pandas as pd


def _to_int(value):
    """
    Excel 값이 939, 939.0, '939', '4,090' 등으로 들어와도 int로 변환.
    변환 불가능하면 None 반환.
    """
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


def load_disfa_error_ranges(raw_dir: str | Path = "data/raw") -> dict[str, list[tuple[int, int]]]:
    """
    DISFA Error_LOG_Sheet.xls를 읽어서 subject별 오류 frame range를 반환한다.

    Error_LOG_Sheet 구조 예:
    row 3: Frames No. Range
    row 4: Subjects No. | From | To | From | To ...
    row 5~: SN030 | 939 | 962 ...

    반환 예:
    {
        "SN030": [(939, 962)],
        "SN029": [(4090, 4543)],
        ...
    }
    """
    raw_dir = Path(raw_dir)
    xls_path = raw_dir / "Error_LOG_Sheet.xls"

    if not xls_path.exists():
        print(f"[WARN] Error_LOG_Sheet not found: {xls_path}")
        return {}

    df = pd.read_excel(xls_path, header=None)

    error_ranges: dict[str, list[tuple[int, int]]] = {}

    # 네가 확인한 구조 기준으로 row 5부터 subject/frame range가 시작됨
    for _, row in df.iloc[5:].iterrows():
        subject = row.iloc[0]

        if pd.isna(subject):
            continue

        subject = str(subject).strip()

        if not re.fullmatch(r"SN\d+", subject):
            continue

        if subject not in error_ranges:
            error_ranges[subject] = []

        # col 1~16이 From/To 쌍으로 반복됨
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
    total_frames = sum((end - start + 1) for ranges in error_ranges.values() for start, end in ranges)

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
    """
    해당 subject/frame이 Error_LOG_Sheet의 오류 구간에 포함되는지 확인.
    """
    ranges = error_ranges.get(subject, [])

    for start, end in ranges:
        if start <= frame <= end:
            return True

    return False