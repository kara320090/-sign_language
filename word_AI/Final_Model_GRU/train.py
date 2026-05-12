"""GRU training entry point (moved into GRU/)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (word_AI) is on sys.path so `from src import ...` works
import os

# 현재 실행 중인 파일의 부모 폴더(Final_Model_GRU)를 기준으로 설정
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent # word_AI 폴더

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# [추가] 설정 파일 로드 로직
import json

CONFIG_PATH = CURRENT_DIR / "experiment_config.json"

def load_run_config():
    if not CONFIG_PATH.exists():
        # 파일이 없을 경우를 대비한 안전장치 (기본값)
        return {"run_name": "Default_Run", "resume": False, "feature_profile": "full"}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_run_config()

# [수정] 결과물 경로를 run_name에 따라 하위 폴더로 분리하여 관리
# 이제 artifacts/GRU_126D/ 와 같이 실험별로 폴더가 생성됩니다.
os.environ["WORD_AI_ARTIFACTS_ROOT"] = str(CURRENT_DIR / "artifacts" / config["run_name"])

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from src.common import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    create_training_summary,
    ensure_tensorflow,
    get_model_dir,
    get_word_data_dir,
    get_result_dir,
    load_dataset,
    save_json,
    slice_feature_channels,
    stratified_split,
    build_feature_datasets,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf = ensure_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_gru_model(input_shape: tuple[int, int], num_classes: int):
    tf = ensure_tensorflow()

    inputs = tf.keras.Input(shape=input_shape, name="word_sequence")
    x = tf.keras.layers.GRU(128, return_sequences=True, name="gru_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.SpatialDropout1D(0.3, name="sdrop_1")(x)
    x = tf.keras.layers.GRU(64, return_sequences=False, name="gru_2")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_1")(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="word_gru_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


def find_overfitting_point(history: dict[str, list[float]]) -> tuple[int | None, str]:
    loss_history = history.get("loss", [])
    val_loss_history = history.get("val_loss", [])

    if not loss_history or not val_loss_history:
        return None, "insufficient history"

    best_val = float("inf")
    overfit_epoch = None

    for epoch_idx, (train_loss, val_loss) in enumerate(zip(loss_history, val_loss_history), start=1):
        if val_loss < best_val:
            best_val = val_loss
        elif train_loss <= min(loss_history[:epoch_idx]) and val_loss > best_val + 1e-3:
            overfit_epoch = epoch_idx
            break

    if overfit_epoch is None:
        return None, "no clear overfitting point detected"

    return overfit_epoch, f"validation loss started to deteriorate around epoch {overfit_epoch}"


def plot_history(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_gru(data_dir: Path, seed: int = DEFAULT_SEED, batch_size: int = 256, 
              feature_indices: list[int] | None = None, resume: bool = False):
    set_global_seed(seed)
    tf = ensure_tensorflow() # tf 로직 사용을 위해 참조

    bundle = load_dataset(data_dir)
    # 차원 조정을 위해 feature_indices 전달
    X = slice_feature_channels(bundle.X, feature_indices=feature_indices)
    y = bundle.y

    train_idx, val_idx = stratified_split(X, y, seed=seed)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_dir = get_model_dir()
    result_dir = get_result_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_feature_datasets(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        seed=seed,
        augment=True,
    )

    model_dir = get_model_dir()
    best_model_path = model_dir / "best_model.keras"

    if resume and best_model_path.exists():
        print(f"\n[Resuming] Loading existing model: {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        # 이어학습 시 기존 학습률의 50%만 사용하여 미세 조정
        tf.keras.backend.set_value(model.optimizer.lr, DEFAULT_LEARNING_RATE * 0.5)
    else:
        print(f"\n[Fresh Start] Building new model with input_dim: {X.shape[-1]}")
        model = build_gru_model(input_shape=(bundle.seq_len, X.shape[-1]), num_classes=bundle.num_classes)

    tf = ensure_tensorflow()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=DEFAULT_PATIENCE,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, DEFAULT_PATIENCE // 3),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=DEFAULT_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = model_dir / "final_model.keras"
    model.save(final_model_path)

    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    save_json(result_dir / "history.json", history_dict)
    plot_history(history_dict, result_dir / "training_history.png")

    overfit_epoch, overfit_message = find_overfitting_point(history_dict)

    summary = {
        "data_dir": str(Path(data_dir).resolve()),
        "seq_len": bundle.seq_len,
        "feature_dim": int(X.shape[-1]),
        "num_classes": int(bundle.num_classes),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "best_model_path": str((model_dir / "best_model.keras").resolve()),
        "final_model_path": str(final_model_path.resolve()),
        "overfit_epoch": overfit_epoch,
        "overfit_message": overfit_message,
        "label_count": int(len(bundle.label_map.get("label_to_idx", {}))),
        "meta": bundle.meta,
    }
    create_training_summary(result_dir / "training_summary.json", **summary)

    print("\nTraining finished.")
    print(f"Best model  : {model_dir / 'best_model.keras'}")
    print(f"Final model : {final_model_path}")
    print(f"History plot : {result_dir / 'training_history.png'}")
    print(f"Summary     : {result_dir / 'training_summary.json'}")
    print(f"Overfit     : {overfit_message}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()

def get_feature_indices(profile: str) -> list[int] | None:
    if profile == "hands":
        return list(range(75, 201))  # 126D
    elif profile == "pose_hands":
        # 양손(126D) + 포즈 핵심 4개점(어깨, 팔꿈치 x,y좌표 8개)
        # MediaPipe Holistic Pose 인덱스 중 11, 12, 13, 14번 활용 예시
        pose_indices = list(range(11*2, 15*2)) 
        hand_indices = list(range(75, 201))
        return pose_indices + hand_indices # 134D
    else:
        return None # full (411D)

def main():
    # 설정 파일에서 읽어온 값 적용
    f_indices = get_feature_indices(config.get("feature_profile", "full"))
    
    print(f"=== Experiment: {config['run_name']} ===")
    print(f"Profile: {config['feature_profile']} | Resume: {config['resume']}")
    
    train_gru(
        data_dir=get_word_data_dir(), 
        seed=DEFAULT_SEED, 
        batch_size=256,
        feature_indices=f_indices,
        resume=config.get("resume", False)
    )

if __name__ == "__main__":
    main()
