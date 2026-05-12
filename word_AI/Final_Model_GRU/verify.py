"""Verification entry point for the word GRU model (moved into GRU/)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (word_AI) is on sys.path
import os
# --- [기존 경로 로직 유지 및 설정 로드 추가] ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent # word_AI 폴더

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
CONFIG_PATH = CURRENT_DIR / "experiment_config.json"

def load_run_config():
    if not CONFIG_PATH.exists():
        return {"run_name": "Default_Run", "resume": False, "feature_profile": "full"}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_run_config()

# [중요] 학습 시와 동일한 하위 폴더를 바라보게 설정
os.environ["WORD_AI_ARTIFACTS_ROOT"] = str(CURRENT_DIR / "artifacts" / config["run_name"])


import argparse
import ctypes
import random
import time
from typing import Any

import numpy as np

from src.common import (
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
    ensure_tensorflow,
    get_result_dir,
    get_word_data_dir,
    load_dataset,
    save_json,
    slice_feature_channels,
    get_model_dir,
)


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf = ensure_tensorflow()
    tf.keras.utils.set_random_seed(seed)


def get_process_memory_mb() -> float:
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    process = kernel32.GetCurrentProcess()

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return float("nan")

    return float(counters.WorkingSetSize) / (1024 * 1024)


def get_gpu_memory_mb() -> dict[str, float]:
    tf = ensure_tensorflow()
    gpu_stats: dict[str, float] = {}

    for device in tf.config.list_physical_devices("GPU"):
        device_name = device.name.split("/")[-1]
        try:
            info = tf.config.experimental.get_memory_info(device_name)
            gpu_stats[device_name] = float(info.get("current", 0)) / (1024 * 1024)
        except Exception:
            gpu_stats[device_name] = float("nan")

    return gpu_stats


def load_best_model(model_path: Path):
    tf = ensure_tensorflow()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    return tf.keras.models.load_model(model_path)


def assert_model_io(model, input_shape: tuple[int, int]) -> dict[str, Any]:
    tf = ensure_tensorflow()
    dummy = tf.zeros((1, input_shape[0], input_shape[1]), dtype=tf.float32)
    prediction = model.predict(dummy, verbose=0)

    if prediction.ndim != 2:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")

    return {
        "dummy_input_shape": list(dummy.shape),
        "prediction_shape": list(prediction.shape),
        "passes": True,
    }


def top_k_report(model, sample: np.ndarray, idx_to_label: dict[str, str], top_k: int = 3) -> dict[str, Any]:
    probs = model.predict(sample, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    top_items = []
    for rank, idx in enumerate(top_indices, start=1):
        label = idx_to_label.get(str(int(idx)), f"class_{idx}")
        top_items.append({
            "rank": rank,
            "class_idx": int(idx),
            "label": label,
            "probability": float(probs[idx]),
        })

    max_prob = float(probs[top_indices[0]])
    decision = "Waiting" if max_prob < DEFAULT_THRESHOLD else idx_to_label.get(str(int(top_indices[0])), f"class_{int(top_indices[0])}")

    return {
        "top_k": top_items,
        "max_probability": max_prob,
        "decision": decision,
    }


def label_mapping_check(y: np.ndarray, idx_to_label: dict[str, str], sample_size: int = 10) -> list[dict[str, Any]]:
    if len(y) == 0:
        return []

    sample_size = min(sample_size, len(y))
    indices = np.linspace(0, len(y) - 1, sample_size, dtype=int)

    checks = []
    for idx in indices:
        label_idx = int(y[idx])
        checks.append({
            "sample_index": int(idx),
            "label_idx": label_idx,
            "label_name": idx_to_label.get(str(label_idx), f"class_{label_idx}"),
            "matches_mapping": str(label_idx) in idx_to_label,
        })

    return checks


def benchmark_latency(model, input_shape: tuple[int, int], repeats: int = 200) -> dict[str, Any]:
    tf = ensure_tensorflow()
    dummy = tf.random.uniform((1, input_shape[0], input_shape[1]), dtype=tf.float32)

    for _ in range(10):
        _ = model.predict(dummy, verbose=0)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(dummy, verbose=0)
    end = time.perf_counter()

    avg_ms = ((end - start) / repeats) * 1000.0
    return {
        "repeats": repeats,
        "average_latency_ms": float(avg_ms),
        "pass_40ms_target": bool(avg_ms <= 40.0),
    }


def verify_gru(data_dir: Path, model_path: Path, seed: int = DEFAULT_SEED, feature_indices: list[int] | None = None) -> dict[str, Any]:
    set_global_seed(seed)

    bundle = load_dataset(data_dir)
    X = slice_feature_channels(bundle.X, feature_indices=feature_indices)
    y = bundle.y
    idx_to_label = bundle.label_map.get("idx_to_label", {})

    model = load_best_model(model_path)

    input_shape = tuple(model.input_shape[1:])
    io_report = assert_model_io(model, input_shape=input_shape)

    tf = ensure_tensorflow()
    sample = tf.convert_to_tensor(X[:1], dtype=tf.float32)
    topk = top_k_report(model, sample, idx_to_label=idx_to_label, top_k=3)

    label_checks = label_mapping_check(y, idx_to_label, sample_size=10)
    latency = benchmark_latency(model, input_shape=input_shape, repeats=200)

    process_memory_mb = get_process_memory_mb()
    gpu_memory_mb = get_gpu_memory_mb()

    results = {
        "model_path": str(model_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "input_shape": list(input_shape),
        "io_report": io_report,
        "top_k_report": topk,
        "label_mapping_checks": label_checks,
        "latency": latency,
        "process_memory_mb": process_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "threshold": DEFAULT_THRESHOLD,
    }

    result_dir = get_result_dir()
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / "verification_report.json", results)

    with open(result_dir / "verification_report.txt", "w", encoding="utf-8") as handle:
        handle.write("Word GRU Verification Report\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Model path: {results['model_path']}\n")
        handle.write(f"Input shape: {results['input_shape']}\n\n")
        handle.write(f"Dummy input check: {io_report}\n\n")
        handle.write(f"Top-3 report: {topk}\n\n")
        handle.write(f"Label mapping checks: {label_checks}\n\n")
        handle.write(f"Latency: {latency}\n\n")
        handle.write(f"Process memory (MB): {process_memory_mb:.2f}\n")
        handle.write(f"GPU memory (MB): {gpu_memory_mb}\n")

    print("\nVerification finished.")
    print(f"Report JSON : {result_dir / 'verification_report.json'}")
    print(f"Report TXT  : {result_dir / 'verification_report.txt'}")
    print(f"Latency(ms) : {latency['average_latency_ms']:.2f}")
    print(f"Decision    : {topk['decision']}")
    print(f"Process RAM : {process_memory_mb:.2f} MB")
    print(f"GPU Memory  : {gpu_memory_mb}")

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(get_word_data_dir()))
    parser.add_argument("--model_path", type=str, default=str(get_model_dir() / "best_model.keras"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()

def get_feature_indices(profile: str) -> list[int] | None:
    if profile == "hands":
        return list(range(75, 201))  # 126D
    elif profile == "pose_hands":
        pose_indices = list(range(11*2, 15*2)) 
        hand_indices = list(range(75, 201))
        return pose_indices + hand_indices # 134D
    else:
        return None # full (411D)
    
def main():
    # 학습 시와 동일한 인덱스 규칙 적용 (get_feature_indices 함수 필요)
    f_indices = get_feature_indices(config.get("feature_profile", "full"))
    
    print(f"=== Verifying Experiment: {config['run_name']} ===")
    
    verify_gru(
        data_dir=get_word_data_dir(), 
        model_path=get_model_dir() / "best_model.keras", 
        seed=DEFAULT_SEED,
        feature_indices=f_indices
    )

if __name__ == "__main__":
    main()
