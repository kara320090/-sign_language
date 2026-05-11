"""Model performance report generator (moved into GRU/)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (word_AI) is on sys.path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer artifacts inside this model folder when running from GRU/
os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "GRU" / "artifacts"))

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.common import (
    DEFAULT_SEED,
    ensure_tensorflow,
    get_result_dir,
    get_word_data_dir,
    load_dataset,
    load_json,
    save_json,
    slice_feature_channels,
    stratified_split,
    get_model_dir,
)


def load_model(model_path: Path):
    tf = ensure_tensorflow()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


# Automatic judgement removed — keep reports neutral for external comparison.


def get_latency_ms(result_dir: Path) -> float | None:
    verify_json = result_dir / "verification_report.json"
    if not verify_json.exists():
        return None

    data = load_json(verify_json)
    latency = data.get("latency", {}).get("average_latency_ms")
    if latency is None:
        return None
    return float(latency)


def top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_label: dict[str, str],
    top_k: int = 2,
) -> list[dict[str, Any]]:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    confused = []
    for i, true_idx in enumerate(labels):
        for j, pred_idx in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            confused.append(
                {
                    "true_idx": int(true_idx),
                    "pred_idx": int(pred_idx),
                    "true_label": idx_to_label.get(str(int(true_idx)), f"class_{int(true_idx)}"),
                    "pred_label": idx_to_label.get(str(int(pred_idx)), f"class_{int(pred_idx)}"),
                    "count": count,
                }
            )

    confused.sort(key=lambda x: x["count"], reverse=True)
    return confused[:top_k]


def build_table_row(
    input_structure: str,
    model_name: str,
    acc: float,
    macro_f1: float,
    weighted_f1: float,
    top1_acc: float,
    strong_precision: float,
    strong_recall: float,
    strong_f1: float,
) -> dict[str, Any]:
    return {
        "입력 구조": input_structure,
        "모델": model_name,
        "Accuracy": round(acc, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Top-1 Accuracy": round(top1_acc, 4),
        "Strong Precision": round(strong_precision, 4),
        "Strong Recall": round(strong_recall, 4),
        "Strong F1": round(strong_f1, 4),
    }


def run_report(
    data_dir: Path,
    model_path: Path,
    output_path: Path,
    seed: int = DEFAULT_SEED,
) -> None:
    bundle = load_dataset(data_dir)
    X = slice_feature_channels(bundle.X)
    y = bundle.y

    model = load_model(model_path)

    # 학습 시점과 같은 분할 전략을 재사용하여 검증셋 지표를 계산한다.
    train_idx, val_idx = stratified_split(X, y, seed=seed)
    X_val = X[val_idx]
    y_val = y[val_idx]

    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    top1_acc = float(np.mean(y_pred == y_val))

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
    weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", 0.0))

    unique_classes = sorted(np.unique(y).tolist())
    strong_idx = int(unique_classes[-1]) if unique_classes else 2
    strong_key = str(strong_idx)

    strong_precision = float(report.get(strong_key, {}).get("precision", 0.0))
    strong_recall = float(report.get(strong_key, {}).get("recall", 0.0))
    strong_f1 = float(report.get(strong_key, {}).get("f1-score", 0.0))

    input_structure = f"{X.shape[1]}F × {X.shape[2]}D"
    row = build_table_row(
        input_structure=input_structure,
        model_name="GRU Baseline",
        acc=float(acc),
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        top1_acc=top1_acc,
        strong_precision=strong_precision,
        strong_recall=strong_recall,
        strong_f1=strong_f1,
    )

    result_dir = get_result_dir()
    latency_ms = get_latency_ms(result_dir)

    idx_to_label = bundle.label_map.get("idx_to_label", {})
    confusion_top2 = top_confused_pairs(y_val, y_pred, idx_to_label=idx_to_label, top_k=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("[성능 검증 결과 표]")
    lines.append(pd.DataFrame([row]).to_string(index=False))
    lines.append("")

    if latency_ms is not None:
        lines.append(f"평균 추론 속도: {latency_ms:.2f} ms")
    else:
        lines.append("평균 추론 속도: verify 결과 파일이 없어 측정값 없음")

    lines.append("")
    lines.append("[혼동되는 단어군 Top-2]")
    if confusion_top2:
        for rank, item in enumerate(confusion_top2, start=1):
            lines.append(
                f"{rank}. {item['true_label']} -> {item['pred_label']} "
                f"(오분류 {item['count']}회)"
            )
    else:
        lines.append("오분류가 감지되지 않았거나 표본이 충분하지 않습니다.")

    lines.append("")
    lines.append("[추가 요약]")
    lines.append(f"Validation samples: {len(y_val)}")
    lines.append(f"Classes: {bundle.num_classes}")

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    metrics_json = {
        "table_row": row,
        "latency_ms": latency_ms,
        "top_confusions": confusion_top2,
        "validation_size": int(len(y_val)),
        "num_classes": int(bundle.num_classes),
        "model_path": str(model_path.resolve()),
        "data_dir": str(data_dir.resolve()),
    }
    save_json(output_path.with_suffix(".json"), metrics_json)

    print(report_text)
    print("\nSaved report:", output_path)
    print("Saved json  :", output_path.with_suffix(".json"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(get_model_dir() / "best_model.keras"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(get_result_dir() / "metrics_table.txt"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    run_report(
        data_dir=Path(args.data_dir),
        model_path=Path(args.model_path),
        output_path=Path(args.output_path),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
