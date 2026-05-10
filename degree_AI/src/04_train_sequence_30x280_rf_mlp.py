import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_NAMES = ["weak", "normal", "strong"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sequence_dataset(data_dir: Path, mode: str):
    x_path = data_dir / f"Xseq_{mode}.npy"
    y_path = data_dir / f"yseq_{mode}.npy"
    meta_path = data_dir / f"metaseq_{mode}.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")

    X = np.load(x_path)
    y = np.load(y_path)
    meta = pd.read_csv(meta_path)

    if X.ndim != 3:
        raise ValueError(f"Xseq must be 3D shape (N, 30, 280), got {X.shape}")

    if X.shape[1] != 30 or X.shape[2] != 280:
        print(f"[WARN] Expected X shape (N, 30, 280), got {X.shape}")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    if len(meta) != len(y):
        print(f"[WARN] meta length and y length differ: {len(meta)} vs {len(y)}")

    return X, y, meta


def flatten_sequence(X: np.ndarray) -> np.ndarray:
    """
    sklearn RF/MLP는 3D 입력을 직접 받지 못하므로
    (N, 30, 280) -> (N, 8400)으로 펼침.
    """
    return X.reshape(X.shape[0], -1).astype(np.float32)


def split_train_val(X_flat, y, meta, test_size=0.2, random_state=42):
    """
    subject 기준으로 train/val 분리.
    같은 사람의 프레임이 train과 val에 동시에 들어가면 과대평가될 수 있으므로
    subject group split을 우선 사용.
    """
    if "subject" in meta.columns:
        groups = meta["subject"].values

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, val_idx = next(splitter.split(X_flat, y, groups=groups))

        return (
            X_flat[train_idx],
            X_flat[val_idx],
            y[train_idx],
            y[val_idx],
            train_idx,
            val_idx,
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_flat,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val, None, None


def oversample_training_data(X_train, y_train, random_state=42):
    """
    MLPClassifier는 class_weight를 직접 쓰기 어렵기 때문에
    학습 데이터에서 클래스별 개수를 맞추는 간단한 랜덤 오버샘플링 수행.
    """
    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    sampled_indices = []

    for cls in classes:
        cls_idx = np.where(y_train == cls)[0]
        sampled = rng.choice(cls_idx, size=max_count, replace=True)
        sampled_indices.append(sampled)

    sampled_indices = np.concatenate(sampled_indices)
    rng.shuffle(sampled_indices)

    return X_train[sampled_indices], y_train[sampled_indices]


def build_model(model_type: str, random_state: int = 42):
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        return model

    if model_type == "mlp":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 256, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=128,
                        learning_rate_init=1e-3,
                        max_iter=120,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=12,
                        random_state=random_state,
                        verbose=True,
                    ),
                ),
            ]
        )
        return model

    raise ValueError("model_type must be 'rf' or 'mlp'")


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(LABEL_NAMES)))
    ax.set_yticks(np.arange(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_and_save(
    model,
    X_val,
    y_val,
    model_type: str,
    out_dir: Path,
    mode: str,
):
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)

    strong_precision = precision_score(y_val, y_pred, labels=[2], average="macro", zero_division=0)
    strong_recall = recall_score(y_val, y_pred, labels=[2], average="macro", zero_division=0)
    strong_f1 = f1_score(y_val, y_pred, labels=[2], average="macro", zero_division=0)

    report_text = classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
    )

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    print("\nClassification Report")
    print(report_text)

    print("\nConfusion Matrix")
    print(cm)

    metrics = {
        "mode": mode,
        "model_type": model_type,
        "input_shape": "30x280",
        "flattened_dim": 30 * 280,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "strong_precision": float(strong_precision),
        "strong_recall": float(strong_recall),
        "strong_f1": float(strong_f1),
    }

    report_path = out_dir / f"results_seq_30x280_{model_type}.txt"
    metrics_path = out_dir / f"metrics_seq_30x280_{model_type}.json"
    cm_csv_path = out_dir / f"confusion_matrix_seq_30x280_{model_type}.csv"
    cm_png_path = out_dir / f"confusion_matrix_seq_30x280_{model_type}.png"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write(report_text)
        f.write("\n\nConfusion Matrix\n")
        f.write(str(cm))
        f.write("\n\nMetrics\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        cm_csv_path,
        encoding="utf-8-sig",
    )

    plot_confusion_matrix(
        cm,
        cm_png_path,
        title=f"30x280 Sequence {model_type.upper()} Confusion Matrix",
    )

    print(f"\nSaved report: {report_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix CSV: {cm_csv_path}")
    print(f"Saved confusion matrix image: {cm_png_path}")

    return metrics


def train_one_model(
    data_dir: Path,
    model_dir: Path,
    result_dir: Path,
    mode: str,
    model_type: str,
    test_size: float,
    random_state: int,
):
    ensure_dir(model_dir)
    ensure_dir(result_dir)

    print("=" * 80)
    print(f"Training 30x280 sequence model: {model_type}")
    print("=" * 80)

    X, y, meta = load_sequence_dataset(data_dir, mode)

    print(f"X original shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_flat = flatten_sequence(X)
    print(f"X flattened shape: {X_flat.shape}")

    print("\nLabel counts:")
    label_counts = pd.Series(y).value_counts().sort_index()
    for label_idx, count in label_counts.items():
        label_name = LABEL_NAMES[int(label_idx)]
        print(f"  {label_idx} ({label_name}): {count}")

    X_train, X_val, y_train, y_val, train_idx, val_idx = split_train_val(
        X_flat,
        y,
        meta,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    print("\nTrain label counts:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    for label_idx, count in train_counts.items():
        print(f"  {label_idx} ({LABEL_NAMES[int(label_idx)]}): {count}")

    print("\nVal label counts:")
    val_counts = pd.Series(y_val).value_counts().sort_index()
    for label_idx, count in val_counts.items():
        print(f"  {label_idx} ({LABEL_NAMES[int(label_idx)]}): {count}")

    if model_type == "mlp":
        print("\n[INFO] Applying random oversampling for MLP training data.")
        X_train_fit, y_train_fit = oversample_training_data(
            X_train,
            y_train,
            random_state=random_state,
        )

        print("Balanced train label counts:")
        balanced_counts = pd.Series(y_train_fit).value_counts().sort_index()
        for label_idx, count in balanced_counts.items():
            print(f"  {label_idx} ({LABEL_NAMES[int(label_idx)]}): {count}")
    else:
        X_train_fit, y_train_fit = X_train, y_train

    model = build_model(model_type=model_type, random_state=random_state)

    print(f"\n[TRAIN] Fitting {model_type} model...")
    model.fit(X_train_fit, y_train_fit)

    print(f"\n[EVAL] Evaluating {model_type} model...")
    metrics = evaluate_and_save(
        model=model,
        X_val=X_val,
        y_val=y_val,
        model_type=model_type,
        out_dir=result_dir,
        mode=mode,
    )

    model_path = model_dir / f"degree_seq_30x280_{mode}_{model_type}.joblib"

    bundle = {
        "model": model,
        "mode": mode,
        "model_type": model_type,
        "label_names": LABEL_NAMES,
        "input_shape": [30, 280],
        "feature_type": "30 frames x 280D sequence feature",
        "feature_detail": "16 summary + 132 normalized landmark + 132 delta",
        "requires_flatten": True,
        "flattened_dim": 30 * 280,
        "metrics": metrics,
    }

    joblib.dump(bundle, model_path)
    print(f"\nSaved model: {model_path}")

    return metrics


def save_summary(all_metrics: list[dict], result_dir: Path):
    if not all_metrics:
        return

    df = pd.DataFrame(all_metrics)
    summary_path = result_dir / "summary_seq_30x280_models.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved summary: {summary_path}")
    print("\nSummary:")
    print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--result_dir", type=str, default="results/degree_seq_30x280")
    parser.add_argument("--mode", type=str, default="anger", choices=["anger", "overall"])
    parser.add_argument("--model_type", type=str, default="all", choices=["rf", "mlp", "all"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    result_dir = Path(args.result_dir)

    all_metrics = []

    if args.model_type in ["rf", "all"]:
        metrics = train_one_model(
            data_dir=data_dir,
            model_dir=model_dir,
            result_dir=result_dir,
            mode=args.mode,
            model_type="rf",
            test_size=args.test_size,
            random_state=args.random_state,
        )
        all_metrics.append(metrics)

    if args.model_type in ["mlp", "all"]:
        metrics = train_one_model(
            data_dir=data_dir,
            model_dir=model_dir,
            result_dir=result_dir,
            mode=args.mode,
            model_type="mlp",
            test_size=args.test_size,
            random_state=args.random_state,
        )
        all_metrics.append(metrics)

    save_summary(all_metrics, result_dir)


if __name__ == "__main__":
    main()