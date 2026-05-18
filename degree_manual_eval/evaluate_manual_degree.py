import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_CSV = INPUT_DIR / "degree_aihub_summary.csv"

COMBINED_MANUAL_CSV = OUTPUT_DIR / "combined_manual_degree_labels.csv"
MERGED_CSV = OUTPUT_DIR / "combined_manual_degree_with_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "manual_degree_eval_metrics.json"
REPORT_TXT = OUTPUT_DIR / "manual_degree_eval_report.txt"
CONFUSION_MATRIX_CSV = OUTPUT_DIR / "manual_degree_confusion_matrix.csv"
MISMATCH_CSV = OUTPUT_DIR / "manual_degree_mismatches.csv"

VALID_LABELS = {"weak", "normal", "strong"}
LABEL_ORDER = ["weak", "normal", "strong"]


def normalize_degree(value: Any) -> str:
    value = str(value).strip().lower()

    mapping = {
        "약함": "weak",
        "약": "weak",
        "weak": "weak",
        "w": "weak",

        "보통": "normal",
        "중간": "normal",
        "normal": "normal",
        "n": "normal",

        "강함": "strong",
        "강": "strong",
        "strong": "strong",
        "s": "strong",
    }

    return mapping.get(value, value)


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {col.lower().strip(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        f"필수 컬럼을 찾지 못했습니다. 후보={candidates}, 실제 컬럼={list(df.columns)}"
    )


def load_manual_label_files() -> pd.DataFrame:
    manual_files = sorted(INPUT_DIR.glob("manual*.csv"))

    if not manual_files:
        raise FileNotFoundError(
            f"수동 라벨 CSV가 없습니다. {INPUT_DIR} 안에 manual로 시작하는 csv를 넣으세요."
        )

    frames = []

    for file_path in manual_files:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

        sequence_col = find_column(
            df,
            ["sequence_name", "sequence", "video_uid", "name", "id"]
        )

        manual_col = find_column(
            df,
            ["manual_degree", "label", "degree", "manual_label"]
        )

        memo_col = None
        for candidate in ["memo", "note", "comment", "비고"]:
            if candidate in df.columns:
                memo_col = candidate
                break

        out = pd.DataFrame()
        out["sequence_name"] = df[sequence_col].astype(str).str.strip()
        out["manual_degree"] = df[manual_col].apply(normalize_degree)
        out["source_file"] = file_path.name

        if memo_col:
            out["memo"] = df[memo_col].fillna("").astype(str)
        else:
            out["memo"] = ""

        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)

    combined = combined[combined["sequence_name"].notna()]
    combined = combined[combined["sequence_name"].astype(str).str.strip() != ""]

    invalid = combined[~combined["manual_degree"].isin(VALID_LABELS)]
    if not invalid.empty:
        invalid_path = OUTPUT_DIR / "invalid_manual_labels.csv"
        invalid.to_csv(invalid_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            f"manual_degree 값이 weak/normal/strong이 아닌 행이 있습니다. 확인 파일: {invalid_path}"
        )

    duplicated = combined[combined.duplicated("sequence_name", keep=False)]
    if not duplicated.empty:
        duplicated_path = OUTPUT_DIR / "duplicated_manual_labels.csv"
        duplicated.to_csv(duplicated_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            f"중복 sequence_name이 있습니다. 확인 파일: {duplicated_path}"
        )

    combined.to_csv(COMBINED_MANUAL_CSV, index=False, encoding="utf-8-sig")
    return combined


def load_prediction_file() -> pd.DataFrame:
    if not PREDICTION_CSV.exists():
        raise FileNotFoundError(
            f"degree_AI 예측 summary 파일이 없습니다: {PREDICTION_CSV}"
        )

    df = pd.read_csv(PREDICTION_CSV, encoding="utf-8-sig")

    sequence_col = find_column(
        df,
        ["sequence_name", "sequence", "video_uid", "name", "id"]
    )

    pred_col = find_column(
        df,
        ["pred_degree", "degree", "prediction", "pred_label", "degree_pred"]
    )

    out = pd.DataFrame()
    out["sequence_name"] = df[sequence_col].astype(str).str.strip()
    out["pred_degree"] = df[pred_col].apply(normalize_degree)

    optional_columns = [
        "degree_ko",
        "confidence",
        "prob_weak",
        "prob_normal",
        "prob_strong",
        "status",
        "used_frames",
        "frame_count",
    ]

    for col in optional_columns:
        if col in df.columns:
            out[col] = df[col]

    invalid = out[~out["pred_degree"].isin(VALID_LABELS)]
    if not invalid.empty:
        invalid_path = OUTPUT_DIR / "invalid_prediction_labels.csv"
        invalid.to_csv(invalid_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            f"pred_degree 값이 weak/normal/strong이 아닌 행이 있습니다. 확인 파일: {invalid_path}"
        )

    duplicated = out[out.duplicated("sequence_name", keep=False)]
    if not duplicated.empty:
        duplicated_path = OUTPUT_DIR / "duplicated_predictions.csv"
        duplicated.to_csv(duplicated_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            f"예측 결과에 중복 sequence_name이 있습니다. 확인 파일: {duplicated_path}"
        )

    return out


def evaluate(merged: pd.DataFrame) -> Dict[str, Any]:
    y_true = merged["manual_degree"].tolist()
    y_pred = merged["pred_degree"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_df = pd.DataFrame(
        cm,
        index=[f"manual_{label}" for label in LABEL_ORDER],
        columns=[f"pred_{label}" for label in LABEL_ORDER],
    )
    cm_df.to_csv(CONFUSION_MATRIX_CSV, encoding="utf-8-sig")

    mismatches = merged[merged["manual_degree"] != merged["pred_degree"]].copy()
    mismatches.to_csv(MISMATCH_CSV, index=False, encoding="utf-8-sig")

    manual_counts = merged["manual_degree"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    pred_counts = merged["pred_degree"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()

    metrics = {
        "total": int(len(merged)),
        "correct": int((merged["manual_degree"] == merged["pred_degree"]).sum()),
        "incorrect": int((merged["manual_degree"] != merged["pred_degree"]).sum()),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "manual_label_counts": {k: int(v) for k, v in manual_counts.items()},
        "prediction_label_counts": {k: int(v) for k, v in pred_counts.items()},
        "classification_report": report_dict,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def build_report(
    manual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    merged: pd.DataFrame,
    metrics: Dict[str, Any],
) -> str:
    missing_pred = sorted(set(manual_df["sequence_name"]) - set(pred_df["sequence_name"]))
    unused_pred = sorted(set(pred_df["sequence_name"]) - set(manual_df["sequence_name"]))

    report = f"""
degree_AI 수동 라벨 기반 성능검증 리포트
========================================

입력 파일:
- 수동 라벨 폴더: {INPUT_DIR}
- degree_AI 예측 파일: {PREDICTION_CSV}

데이터 병합 결과:
- 수동 라벨 수: {len(manual_df)}
- 예측 결과 수: {len(pred_df)}
- 병합 성공 수: {len(merged)}
- 수동 라벨은 있으나 예측 누락: {len(missing_pred)}
- 예측은 있으나 수동 라벨 없음: {len(unused_pred)}

성능 지표:
- Accuracy: {metrics["accuracy"]:.4f}
- Macro F1: {metrics["macro_f1"]:.4f}
- Weighted F1: {metrics["weighted_f1"]:.4f}
- Correct: {metrics["correct"]}
- Incorrect: {metrics["incorrect"]}

수동 라벨 분포:
{json.dumps(metrics["manual_label_counts"], ensure_ascii=False, indent=2)}

모델 예측 분포:
{json.dumps(metrics["prediction_label_counts"], ensure_ascii=False, indent=2)}

생성 파일:
- {COMBINED_MANUAL_CSV}
- {MERGED_CSV}
- {METRICS_JSON}
- {CONFUSION_MATRIX_CSV}
- {MISMATCH_CSV}
- {REPORT_TXT}

해석:
- Accuracy는 전체 정답률을 의미한다.
- Macro F1은 weak, normal, strong 각 라벨을 동일한 비중으로 평가한다.
- Weighted F1은 라벨별 샘플 수를 반영한 F1-score이다.
- Confusion Matrix는 어떤 라벨에서 오분류가 발생했는지 확인하기 위한 표이다.
""".strip()

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manual_df = load_manual_label_files()
    pred_df = load_prediction_file()

    merged = pd.merge(
        manual_df,
        pred_df,
        on="sequence_name",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "수동 라벨과 예측 결과가 sequence_name 기준으로 하나도 병합되지 않았습니다."
        )

    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")

    metrics = evaluate(merged)
    report = build_report(manual_df, pred_df, merged, metrics)

    print(report)


if __name__ == "__main__":
    main()