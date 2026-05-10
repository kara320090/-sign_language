import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["weak", "normal", "strong"]


def normalize_label(value):
    if pd.isna(value):
        return ""

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


def load_label_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = str(path)

    if "manual_degree" not in df.columns:
        raise ValueError(f"manual_degree column not found: {path}")

    if "pred_degree" not in df.columns:
        raise ValueError(f"pred_degree column not found: {path}")

    df["manual_degree"] = df["manual_degree"].apply(normalize_label)
    df["pred_degree"] = df["pred_degree"].apply(normalize_label)

    return df


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_csv", required=True)
    parser.add_argument("--teammate_csv", required=True)
    parser.add_argument("--out_dir", default="outputs/manual_degree_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    my_df = load_label_file(Path(args.my_csv))
    teammate_df = load_label_file(Path(args.teammate_csv))

    df = pd.concat([my_df, teammate_df], ignore_index=True)

    # manual_degree가 비어 있는 행 제거
    df = df[df["manual_degree"].isin(LABELS)].copy()
    df = df[df["pred_degree"].isin(LABELS)].copy()

    if df.empty:
        raise RuntimeError("No valid labeled rows found. Check manual_degree values.")

    y_true = df["manual_degree"].tolist()
    y_pred = df["pred_degree"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    df["is_correct"] = df["manual_degree"] == df["pred_degree"]

    combined_csv = out_dir / "combined_manual_degree_labels.csv"
    mismatch_csv = out_dir / "manual_degree_mismatches.csv"
    cm_csv = out_dir / "manual_degree_confusion_matrix.csv"
    report_txt = out_dir / "manual_degree_eval_report.txt"
    metrics_json = out_dir / "manual_degree_eval_metrics.json"

    df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    df[df["is_correct"] == False].to_csv(mismatch_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(cm_csv, encoding="utf-8-sig")

    metrics = {
        "num_samples": int(len(df)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "label_order": LABELS,
        "manual_label_counts": df["manual_degree"].value_counts().to_dict(),
        "pred_label_counts": df["pred_degree"].value_counts().to_dict(),
        "correct_count": int(df["is_correct"].sum()),
        "wrong_count": int((~df["is_correct"]).sum()),
    }

    save_json(metrics_json, metrics)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Manual Degree Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report\n")
        f.write(report)
        f.write("\n\nConfusion Matrix\n")
        f.write(str(cm))
        f.write("\n\nManual label counts\n")
        f.write(str(df["manual_degree"].value_counts()))
        f.write("\n\nPrediction label counts\n")
        f.write(str(df["pred_degree"].value_counts()))

    print("[DONE] Manual degree evaluation finished.")
    print(f"Samples      : {len(df)}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Correct      : {df['is_correct'].sum()}")
    print(f"Wrong        : {(~df['is_correct']).sum()}")
    print()
    print(f"Combined CSV : {combined_csv}")
    print(f"Mismatch CSV : {mismatch_csv}")
    print(f"Report TXT   : {report_txt}")
    print(f"Metrics JSON : {metrics_json}")
    print(f"CM CSV       : {cm_csv}")


if __name__ == "__main__":
    main()