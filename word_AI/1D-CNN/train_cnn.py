"""1D-CNN training entry point for word-level sign language modeling."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("WORD_AI_ARTIFACTS_ROOT", str(ROOT / "1D-CNN" / "artifacts"))

from cnn_common import (  # noqa: E402
    CNN_BATCH_SIZE,
    CNN_EARLY_STOP_PATIENCE,
    CNN_EPOCHS,
    CNN_FEATURE_DIM,
    CNN_LR,
    CNN_MIN_LR,
    CNN_REDUCE_PATIENCE,
    CNN_SEED,
    build_cnn_datasets,
    load_core_indices,
    load_cnn_dataset,
    safe_stratified_split,
    save_cnn_json,
    save_cnn_summary,
)
from src.common import ensure_tensorflow, get_model_dir, get_result_dir, get_word_data_dir  # noqa: E402

tf = ensure_tensorflow()


DEFAULT_WARMUP_EPOCHS = 0
FeatureProfile = Literal["baseline216", "body_hands201", "hands126", "core126"]
LossMode = Literal["sparse", "focal"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf = ensure_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def maybe_limit_samples(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(X):
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    indices.sort()
    return X[indices], y[indices]


def random_frame_shift(sequence, max_shift: int = 1):
    tf = ensure_tensorflow()
    seq_len = tf.shape(sequence)[0]
    shift = tf.random.uniform([], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)
    idx = tf.range(seq_len, dtype=tf.int32) + shift
    idx = tf.clip_by_value(idx, 0, seq_len - 1)
    return tf.gather(sequence, idx)


def augment_cnn(sequence, label, num_classes: int):
    tf = ensure_tensorflow()
    sequence = tf.cast(sequence, tf.float32)
    sequence = random_frame_shift(sequence, max_shift=1)
    # Keep label as integer (sparse_categorical_crossentropy)
    return sequence, label


def prepare_eval_item(sequence, label, num_classes: int):
    tf = ensure_tensorflow()
    sequence = tf.cast(sequence, tf.float32)
    # Keep label as integer (sparse_categorical_crossentropy)
    return sequence, label


def resolve_profile_indices(profile: FeatureProfile | None) -> list[int] | None:
    if profile is None:
        return None
    if profile == "baseline216":
        return list(range(216))
    if profile == "body_hands201":
        return list(range(201))
    if profile == "hands126":
        return list(range(75, 201))
    if profile == "core126":
        return load_core_indices().astype(int).tolist()
    raise ValueError(f"Unsupported feature profile: {profile}")


def build_loss(
    loss_mode: LossMode,
    focal_gamma: float,
    focal_alpha: float,
    label_smoothing: float,
):
    tf = ensure_tensorflow()
    if loss_mode == "sparse":
        if label_smoothing > 0.0:
            print("Warning: label_smoothing is not supported in this TF build for sparse CE. Ignoring it.")
        return tf.keras.losses.SparseCategoricalCrossentropy()

    def sparse_focal_loss(y_true, y_pred):
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1], dtype=tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred_clipped), axis=-1)
        pt = tf.reduce_sum(y_true_oh * y_pred_clipped, axis=-1)
        focal_term = tf.pow(1.0 - pt, focal_gamma)
        loss = focal_alpha * focal_term * ce
        return tf.reduce_mean(loss)

    return sparse_focal_loss


def compute_class_weight(y_train: np.ndarray, power: float = 0.5) -> dict[int, float]:
    classes, counts = np.unique(y_train, return_counts=True)
    inv = 1.0 / np.power(counts.astype(np.float64), power)
    inv = inv / np.mean(inv)
    return {int(c): float(w) for c, w in zip(classes, inv)}


def build_cnn_model(
    input_shape: tuple[int, int],
    num_classes: int,
    loss_mode: LossMode = "sparse",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_smoothing: float = 0.0,
):
    tf = ensure_tensorflow()

    inputs = tf.keras.Input(shape=input_shape, name="word_sequence")

    x = tf.keras.layers.Conv1D(64, 3, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = tf.keras.layers.Conv1D(128, 3, padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense1",
    )(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout1")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="word_1dcnn_classifier")
    loss_obj = build_loss(
        loss_mode=loss_mode,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        label_smoothing=label_smoothing,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_LR),
        loss=loss_obj,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


class LearningRateRecorder(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.learning_rates: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        tf = ensure_tensorflow()
        learning_rate = self.model.optimizer.learning_rate
        if hasattr(learning_rate, "numpy"):
            value = float(learning_rate.numpy())
        else:
            value = float(tf.keras.backend.get_value(learning_rate))
        self.learning_rates.append(value)


class WarmupLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, target_lr: float, warmup_epochs: int = DEFAULT_WARMUP_EPOCHS) -> None:
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return
        tf = ensure_tensorflow()
        warmup_lr = self.target_lr * float(epoch + 1) / float(self.warmup_epochs)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)


def plot_history(history: dict[str, list[float]], learning_rates: list[float], out_path: Path) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(epochs, history.get("loss", []), label="train_loss")
    axes[0].plot(epochs, history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.get("accuracy", []), label="train_acc")
    axes[1].plot(epochs, history.get("val_accuracy", []), label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs[: len(learning_rates)], learning_rates, label="learning_rate", color="tab:green")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_cnn(
    data_dir: Path,
    seed: int = CNN_SEED,
    batch_size: int = CNN_BATCH_SIZE,
    epochs: int = CNN_EPOCHS,
    feature_indices: list[int] | None = None,
    feature_profile: FeatureProfile | None = None,
    loss_mode: LossMode = "sparse",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_smoothing: float = 0.0,
    use_class_weight: bool = False,
    class_weight_power: float = 0.5,
    max_samples: int | None = None,
    run_name: str | None = None,
):
    set_global_seed(seed)

    selected_indices = feature_indices
    if selected_indices is None:
        selected_indices = resolve_profile_indices(feature_profile)

    bundle = load_cnn_dataset(data_dir, feature_indices=selected_indices)
    X = bundle.X
    y = bundle.y

    X, y = maybe_limit_samples(X, y, max_samples=max_samples, seed=seed)

    train_idx, val_idx = safe_stratified_split(X, y, seed=seed)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_ds, val_ds = build_cnn_datasets(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=bundle.num_classes,
        batch_size=batch_size,
        seed=seed,
    )

    model = build_cnn_model(
        input_shape=(bundle.seq_len, X.shape[-1]),
        num_classes=bundle.num_classes,
        loss_mode=loss_mode,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        label_smoothing=label_smoothing,
    )

    model_dir_base = get_model_dir()
    result_dir_base = get_result_dir()
    model_dir = model_dir_base / run_name if run_name else model_dir_base
    result_dir = result_dir_base / run_name if run_name else result_dir_base
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    class_weight = None
    if use_class_weight:
        class_weight = compute_class_weight(y_train, power=class_weight_power)
        print(f"Using class_weight with {len(class_weight)} classes (power={class_weight_power})")

    tf = ensure_tensorflow()
    lr_recorder = LearningRateRecorder()
    callbacks = [
        WarmupLearningRate(target_lr=CNN_LR, warmup_epochs=DEFAULT_WARMUP_EPOCHS),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_cnn_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CNN_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=CNN_REDUCE_PATIENCE,
            min_lr=CNN_MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        lr_recorder,
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    final_model_path = model_dir / "final_cnn_model.keras"
    model.save(final_model_path)

    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    save_cnn_json(result_dir / "history_cnn.json", history_dict)
    plot_history(history_dict, lr_recorder.learning_rates, result_dir / "training_history_cnn.png")

    summary = {
        "model": "1D-CNN Baseline",
        "data_dir": str(data_dir.resolve()),
        "seq_len": int(bundle.seq_len),
        "feature_dim": int(X.shape[-1]),
        "num_classes": int(bundle.num_classes),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "best_model_path": str((model_dir / "best_cnn_model.keras").resolve()),
        "final_model_path": str(final_model_path.resolve()),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "seed": int(seed),
        "feature_profile": feature_profile,
        "feature_indices": selected_indices,
        "loss_mode": loss_mode,
        "focal_gamma": float(focal_gamma),
        "focal_alpha": float(focal_alpha),
        "label_smoothing": float(label_smoothing),
        "use_class_weight": bool(use_class_weight),
        "class_weight_power": float(class_weight_power),
        "run_name": run_name,
        "max_samples": int(max_samples) if max_samples else None,
    }
    save_cnn_summary(result_dir / "training_summary_cnn.json", **summary)

    print("\nCNN training finished.")
    print(f"Best model  : {model_dir / 'best_cnn_model.keras'}")
    print(f"Final model : {final_model_path}")
    print(f"History plot: {result_dir / 'training_history_cnn.png'}")
    print(f"Summary     : {result_dir / 'training_summary_cnn.json'}")


def parse_feature_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        return None
    return [int(v) for v in values]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument("--batch_size", type=int, default=CNN_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=CNN_SEED)
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS)
    parser.add_argument(
        "--feature_indices",
        type=str,
        default=None,
        help="Comma-separated feature indices. Omit to use GRU-consistent full channels.",
    )
    parser.add_argument(
        "--feature_profile",
        type=str,
        default="baseline216",
        choices=["baseline216", "body_hands201", "hands126", "core126"],
        help="Predefined feature profile. Ignored when --feature_indices is provided.",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="sparse",
        choices=["sparse", "focal"],
        help="Loss mode: sparse cross-entropy or focal loss.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        help="Apply inverse-frequency class weighting.",
    )
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional subfolder name under artifacts/models and artifacts/results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_cnn(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        feature_indices=parse_feature_indices(args.feature_indices),
        feature_profile=args.feature_profile,
        loss_mode=args.loss_mode,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing,
        use_class_weight=args.use_class_weight,
        class_weight_power=args.class_weight_power,
        max_samples=args.max_samples,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
