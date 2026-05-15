from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


INPUT_SHAPE = (30, 120)
BATCH_SIZE = 64
LATENCY_WARMUP = 20
LATENCY_REPEATS = 200
SAMPLE_PREDICTIONS = 50
TOP_CONFUSIONS = 20
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cnn"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODEL_PATH = MODEL_DIR / "cnn_best.keras"
HISTORY_PATH = RESULTS_DIR / "history.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_TOP_PATH = RESULTS_DIR / "confusion_top.csv"
SAMPLE_PREDICTIONS_PATH = RESULTS_DIR / "sample_predictions.csv"
LEARNING_CURVE_PATH = RESULTS_DIR / "learning_curve.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TOP_CONFUSIONS_PATH = RESULTS_DIR / "top_confusions.png"


def configure_korean_font():
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "AppleGothic",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"Using matplotlib font: {font_name}")
            return

    plt.rcParams["axes.unicode_minus"] = False
    print(
        "Warning: Korean font not found. "
        "PNG labels may be broken. Install NanumGothic or use Windows Malgun Gothic."
    )


def make_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_validation():
    x_path = VALIDATION_DIR / "X_validation.npy"
    y_path = VALIDATION_DIR / "y_validation.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Validation split missing. Run `python data/prepare_split.py` first.")
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    assert X.shape[1:] == INPUT_SHAPE
    assert y.ndim == 1
    return X, y


def load_classes():
    classes_path = VALIDATION_DIR / "classes.npy"
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file missing: {classes_path}")
    classes = np.load(classes_path, allow_pickle=True)
    return np.array([str(label).strip() for label in classes], dtype=object)


def load_optional_test():
    x_path = TEST_DIR / "X_test.npy"
    y_path = TEST_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        return None
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    assert X.shape[1:] == INPUT_SHAPE
    assert y.ndim == 1
    return X, y


def measure_latency(model):
    sample = np.zeros((1, *INPUT_SHAPE), dtype=np.float32)

    for _ in range(LATENCY_WARMUP):
        _ = model.predict(sample, verbose=0)

    times = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        _ = model.predict(sample, verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times, dtype=np.float64)
    return {
        "latency_avg_ms": float(np.mean(times)),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
        "latency_repeats": int(LATENCY_REPEATS),
    }


def evaluate_split(model, split_name, X, y, note, latency):
    loss, keras_acc, top3 = model.evaluate(make_dataset(X, y), verbose=0)
    pred_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    pred = np.argmax(pred_prob, axis=1)

    metrics = {
        "split": split_name,
        "loss": float(loss),
        "accuracy": float(accuracy_score(y, pred)),
        "keras_accuracy": float(keras_acc),
        "top3_accuracy": float(top3),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "num_samples": int(len(y)),
        "note": note,
        **latency,
    }
    return metrics, pred_prob, pred


def save_classification_report(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes.tolist(),
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    df.to_csv(CLASSIFICATION_REPORT_PATH, index=False, encoding="utf-8-sig")
    print("Saved:", CLASSIFICATION_REPORT_PATH)


def build_confusion_top(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    for true_idx in labels:
        for pred_idx in labels:
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            rows.append(
                {
                    "true_label": true_idx,
                    "true_text": classes[true_idx],
                    "pred_label": pred_idx,
                    "pred_text": classes[pred_idx],
                    "count": count,
                }
            )

    columns = ["true_label", "true_text", "pred_label", "pred_text", "count"]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("count", ascending=False).head(TOP_CONFUSIONS)
    df.to_csv(CONFUSION_TOP_PATH, index=False, encoding="utf-8-sig")
    print("Saved:", CONFUSION_TOP_PATH)
    return cm, df


def save_sample_predictions(y_true, pred_prob, y_pred, classes):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_count = min(SAMPLE_PREDICTIONS, len(y_true))
    sample_indices = rng.choice(len(y_true), size=sample_count, replace=False)
    sample_indices = np.sort(sample_indices)

    rows = []
    for sample_index in sample_indices:
        probs = pred_prob[sample_index]
        top_idx = np.argsort(probs)[-3:][::-1]
        true_label = int(y_true[sample_index])
        pred_label = int(y_pred[sample_index])

        row = {
            "sample_index": int(sample_index),
            "true_label": true_label,
            "true_text": classes[true_label],
            "pred_label": pred_label,
            "pred_text": classes[pred_label],
            "confidence": float(probs[pred_label]),
            "correct": bool(true_label == pred_label),
        }

        for rank, idx in enumerate(top_idx, start=1):
            idx = int(idx)
            row[f"top{rank}_label"] = idx
            row[f"top{rank}_text"] = classes[idx]
            row[f"top{rank}_prob"] = float(probs[idx])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SAMPLE_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print("Saved:", SAMPLE_PREDICTIONS_PATH)


def save_learning_curve():
    if not HISTORY_PATH.exists():
        print(f"Skipping learning curve; missing: {HISTORY_PATH}")
        return

    history = pd.read_csv(HISTORY_PATH)
    epochs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "loss" in history:
        axes[0].plot(epochs, history["loss"], label="train_loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "accuracy" in history:
        axes[1].plot(epochs, history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(LEARNING_CURVE_PATH, dpi=180)
    plt.close(fig)
    print("Saved:", LEARNING_CURVE_PATH)


def save_confusion_matrix_plot(cm):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close()
    print("Saved:", CONFUSION_MATRIX_PATH)


def save_top_confusions_plot(confusion_top_df):
    plt.figure(figsize=(12, 8))

    if confusion_top_df.empty:
        plt.text(0.5, 0.5, "No misclassifications", ha="center", va="center")
        plt.axis("off")
    else:
        plot_df = confusion_top_df.iloc[::-1].copy()
        plot_df["pair"] = plot_df["true_text"] + " -> " + plot_df["pred_text"]
        plt.barh(plot_df["pair"], plot_df["count"])
        plt.xlabel("Count")
        plt.title(f"Top {len(plot_df)} Confusions")
        plt.tight_layout()

    plt.savefig(TOP_CONFUSIONS_PATH, dpi=180)
    plt.close()
    print("Saved:", TOP_CONFUSIONS_PATH)


def save_validation_details(y_true, pred_prob, y_pred, classes):
    save_classification_report(y_true, y_pred, classes)
    cm, confusion_top_df = build_confusion_top(y_true, y_pred, classes)
    save_sample_predictions(y_true, pred_prob, y_pred, classes)
    save_learning_curve()
    save_confusion_matrix_plot(cm)
    save_top_confusions_plot(confusion_top_df)


def main():
    configure_korean_font()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    X_validation, y_validation = load_validation()
    classes = load_classes()
    latency = measure_latency(model)

    validation_metrics, validation_pred_prob, validation_pred = evaluate_split(
        model,
        "validation",
        X_validation,
        y_validation,
        "internal 20 percent split from current npy data",
        latency,
    )
    save_validation_details(y_validation, validation_pred_prob, validation_pred, classes)
    rows = [validation_metrics]

    test_data = load_optional_test()
    if test_data is not None:
        X_test, y_test = test_data
        test_metrics, _, _ = evaluate_split(
            model,
            "test",
            X_test,
            y_test,
            "external validation data after preprocessing",
            latency,
        )
        rows.append(test_metrics)

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    print(df.to_string(index=False))
    print("\nSaved:", METRICS_PATH)


if __name__ == "__main__":
    main()

