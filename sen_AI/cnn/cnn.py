import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


SEED = 42
INPUT_SHAPE = (30, 120)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cnn"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
MODEL_PATH = MODEL_DIR / "cnn_best.keras"
HISTORY_PATH = RESULTS_DIR / "history.csv"


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_split():
    paths = [
        TRAIN_DIR / "X_train.npy",
        TRAIN_DIR / "y_train.npy",
        VALIDATION_DIR / "X_validation.npy",
        VALIDATION_DIR / "y_validation.npy",
        VALIDATION_DIR / "classes.npy",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Split files are missing. Run `python data/prepare_split.py` first.\n"
            + "\n".join(missing)
        )

    X_train = np.load(TRAIN_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(TRAIN_DIR / "y_train.npy").astype(np.int64)
    X_validation = np.load(VALIDATION_DIR / "X_validation.npy").astype(np.float32)
    y_validation = np.load(VALIDATION_DIR / "y_validation.npy").astype(np.int64)
    classes = np.load(VALIDATION_DIR / "classes.npy", allow_pickle=True)
    num_classes = len(classes)

    assert X_train.shape[1:] == INPUT_SHAPE
    assert X_validation.shape[1:] == INPUT_SHAPE
    assert y_train.ndim == 1
    assert y_validation.ndim == 1
    assert y_train.min() >= 0 and y_train.max() < num_classes
    assert y_validation.min() >= 0 and y_validation.max() < num_classes

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)
    print("classes:", num_classes)

    return X_train, y_train, X_validation, y_validation, num_classes


def make_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


def main():
    set_seeds()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_validation, y_validation, num_classes = load_split()
    train_ds = make_dataset(X_train, y_train, shuffle=True)
    validation_ds = make_dataset(X_validation, y_validation, shuffle=False)

    model = build_model(num_classes)
    model.summary()

    cb = [
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

    print("\nSaved model:", MODEL_PATH)
    print("Saved history:", HISTORY_PATH)


if __name__ == "__main__":
    main()
