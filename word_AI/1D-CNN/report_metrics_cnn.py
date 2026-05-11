"""Team-format metrics report generator for 1D-CNN baseline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "1D-CNN" / "artifacts"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from cnn_common import CNN_SEED, load_cnn_dataset, load_core_indices, safe_stratified_split, save_cnn_json  # noqa: E402
from src.common import ensure_tensorflow, get_model_dir, get_result_dir, get_word_data_dir  # noqa: E402


def load_model(model_path: Path):
    tf = ensure_tensorflow()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Report step only needs inference; compile=False avoids custom loss deserialization issues.
    return tf.keras.models.load_model(model_path, compile=False)


# Automatic judgement removed — decisions should be recorded externally when comparing models.


def run_report(data_dir: Path, model_path: Path, output_path: Path, seed: int = CNN_SEED) -> None:
    # Load model FIRST to get expected input shape
    model = load_model(model_path)
    seq_len, feature_dim = tuple(model.input_shape[1:])
    
    # Match feature slicing to model input dimension.
    # 126D uses curated core indices, others use the first N features.
    if feature_dim == 126:
        feature_indices = load_core_indices()
    else:
        feature_indices = np.arange(feature_dim, dtype=np.int32)
    
    bundle = load_cnn_dataset(data_dir, feature_indices=feature_indices)
    X = bundle.X
    y = bundle.y

    _, val_idx = safe_stratified_split(X, y, seed=seed)
    X_val = X[val_idx]
    y_val = y[val_idx]

    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_val, y_pred))
    cls_report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    macro_f1 = float(cls_report.get("macro avg", {}).get("f1-score", 0.0))
    weighted_f1 = float(cls_report.get("weighted avg", {}).get("f1-score", 0.0))

    tf = ensure_tensorflow()
    dummy = tf.random.uniform((1, X.shape[1], X.shape[2]), dtype=tf.float32)
    start = tf.timestamp()
    for _ in range(200):
        _ = model.predict(dummy, verbose=0)
    latency_ms = float((tf.timestamp() - start).numpy() * 1000.0 / 200.0)

    # Dynamically set input_structure based on actual feature dimension
    input_structure = f"30F × {feature_dim}D"
    if feature_dim == 126:
        input_structure += " (Hands Only)"
    elif feature_dim == 201:
        input_structure += " (Body+Hands)"
    elif feature_dim == 216:
        input_structure += " (Baseline)"
    
    row: dict[str, Any] = {
        "입력 구조": input_structure,
        "모델": "1D-CNN Baseline",
        "Accuracy": round(acc, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
    }

    table = pd.DataFrame([row])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "[1D-CNN 성능 검증 결과 표]",
        table.to_string(index=False),
        "",
        f"Validation samples: {len(y_val)}",
        f"Classes: {bundle.num_classes}",
        f"Average latency(ms): {latency_ms:.2f}",
    ]
    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    save_cnn_json(
        output_path.with_suffix(".json"),
        {
            "table_row": row,
            "validation_size": int(len(y_val)),
            "num_classes": int(bundle.num_classes),
            "model_path": str(model_path.resolve()),
            "data_dir": str(data_dir.resolve()),
            "latency_ms": latency_ms,
            "seed": int(seed),
        },
    )

    print(report_text)
    print(f"\nSaved report: {output_path}")
    print(f"Saved json  : {output_path.with_suffix('.json')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument("--model_path", type=str, default=str(get_model_dir() / "best_cnn_model.keras"))
    parser.add_argument("--output_path", type=str, default=str(get_result_dir() / "cnn_performance_report.txt"))
    parser.add_argument("--seed", type=int, default=CNN_SEED)
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
