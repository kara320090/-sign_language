from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIRS = {
    "lstm": RESULTS_DIR / "lstm",
    "gru": RESULTS_DIR / "gru",
    "cnn": RESULTS_DIR / "cnn",
}
OUTPUT_PATH = RESULTS_DIR / "model_comparison.csv"


def main():
    frames = []

    for model_name, model_dir in MODEL_DIRS.items():
        metrics_path = model_dir / "metrics.csv"
        if not metrics_path.exists():
            print(f"missing: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        df.insert(0, "model", model_name)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No metrics.csv files found. Run report_metrics scripts first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nModel comparison")
    print(comparison.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
