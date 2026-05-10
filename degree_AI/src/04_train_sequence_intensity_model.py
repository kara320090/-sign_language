import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_NAMES = ["weak", "normal", "strong"]


def train_sequence_model(data_dir: Path, model_dir: Path, mode: str, model_type: str) -> None:
    X_path = data_dir / f"Xseq_{mode}.npy"
    y_path = data_dir / f"yseq_{mode}.npy"
    meta_path = data_dir / f"metaseq_{mode}.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"X original shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # sklearn 모델은 3D 입력을 바로 못 받기 때문에 펼침
    # (N, 30, 280) -> (N, 8400)
    X_flat = X.reshape(X.shape[0], -1)

    print(f"X flattened shape: {X_flat.shape}")

    label_counts = pd.Series(y).value_counts().sort_index()
    print("\nLabel counts:")
    for idx, count in label_counts.items():
        print(f"  {idx} ({LABEL_NAMES[int(idx)]}): {count}")

    if meta_path.exists():
        meta = pd.read_csv(meta_path)

        if "subject" in meta.columns:
            groups = meta["subject"].values

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=42
            )

            train_idx, val_idx = next(splitter.split(X_flat, y, groups=groups))
            X_train, X_val = X_flat[train_idx], X_flat[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    elif model_type == "mlp":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=100,
                random_state=42,
                verbose=True
            ))
        ])

    else:
        raise ValueError("model_type must be 'rf' or 'mlp'")

    print(f"\nTraining sequence model: {model_type}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("\nClassification Report")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"degree_seq_{mode}_{model_type}.joblib"

    bundle = {
        "model": model,
        "mode": mode,
        "model_type": model_type,
        "label_names": LABEL_NAMES,
        "input_shape": list(X.shape[1:]),
        "feature_type": "sequence summary16 + landmark132 + delta132",
        "requires_flatten": True
    }

    joblib.dump(bundle, model_path)

    print(f"\nSaved model: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--mode", type=str, default="anger", choices=["anger", "overall"])
    parser.add_argument("--model_type", type=str, default="mlp", choices=["rf", "mlp"])
    args = parser.parse_args()

    train_sequence_model(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        mode=args.mode,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()