import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


POSE_LEN = 25 * 3
HAND_LEN = 21 * 3
FACE_LEN = 70 * 3

DEGREE_MAP_KO_TO_EN = {
    "약함": "weak",
    "약": "weak",
    "weak": "weak",
    "w": "weak",
    "0": "weak",
    "보통": "normal",
    "중간": "normal",
    "normal": "normal",
    "n": "normal",
    "1": "normal",
    "강함": "strong",
    "강": "strong",
    "strong": "strong",
    "s": "strong",
    "2": "strong",
}

WEAK_MODIFIERS = ["약간", "조금", "살짝", "약하게", "조금씩"]
STRONG_MODIFIERS = ["매우", "정말", "엄청", "극도로", "무척", "아주", "진짜"]


def normalize_degree(value: Any, default: str = "normal") -> str:
    text = "" if value is None else str(value).strip().lower()
    if not text:
        return default
    return DEGREE_MAP_KO_TO_EN.get(text, default)


def check_degree_modifier(final_text: str, degree_en: str) -> str:
    final_text = (final_text or "").strip().lower()
    has_weak = any(mod in final_text for mod in WEAK_MODIFIERS)
    has_strong = any(mod in final_text for mod in STRONG_MODIFIERS)

    if degree_en == "weak":
        return "PASS" if has_weak else "FAIL"
    if degree_en == "strong":
        return "PASS" if has_strong else "FAIL"
    return "PASS" if (not has_weak and not has_strong) else "FAIL"


def get_people(data: Dict[str, Any]) -> Dict[str, Any]:
    people = data.get("people", {})
    if isinstance(people, list):
        return people[0] if people and isinstance(people[0], dict) else {}
    if isinstance(people, dict):
        return people
    return {}


def fix_length(arr: Any, target_len: int) -> List[float]:
    if not isinstance(arr, list):
        return [0.0] * target_len
    if len(arr) >= target_len:
        return arr[:target_len]
    return arr + [0.0] * (target_len - len(arr))


def reshape_keypoints(flat_arr: List[float]) -> np.ndarray:
    return np.asarray(flat_arr, dtype=np.float32).reshape(-1, 3)


def get_origin_and_scale(pose_arr: np.ndarray):
    origin = np.array([0.0, 0.0], dtype=np.float32)
    scale = 1.0
    try:
        neck = pose_arr[1]
        r_shoulder = pose_arr[2]
        l_shoulder = pose_arr[5]

        if r_shoulder[2] > 0 and l_shoulder[2] > 0:
            origin = (r_shoulder[:2] + l_shoulder[:2]) / 2.0
            shoulder_width = float(np.linalg.norm(r_shoulder[:2] - l_shoulder[:2]))
            if shoulder_width > 1e-6:
                scale = shoulder_width
        elif neck[2] > 0:
            origin = neck[:2].astype(np.float32)
    except Exception:
        pass

    return origin, scale


def normalize_points(arr: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    out = arr.copy()
    confident = out[:, 2] > 0
    out[~confident, 0:2] = 0.0
    out[confident, 0] = (out[confident, 0] - origin[0]) / scale
    out[confident, 1] = (out[confident, 1] - origin[1]) / scale
    return out


def extract_frame_feature(data: Dict[str, Any]) -> np.ndarray:
    people = get_people(data)
    pose = reshape_keypoints(fix_length(people.get("pose_keypoints_2d", []), POSE_LEN))
    left = reshape_keypoints(fix_length(people.get("hand_left_keypoints_2d", []), HAND_LEN))
    right = reshape_keypoints(fix_length(people.get("hand_right_keypoints_2d", []), HAND_LEN))
    face = reshape_keypoints(fix_length(people.get("face_keypoints_2d", []), FACE_LEN))

    origin, scale = get_origin_and_scale(pose)
    pose = normalize_points(pose, origin, scale)
    left = normalize_points(left, origin, scale)
    right = normalize_points(right, origin, scale)
    face = normalize_points(face, origin, scale)

    return np.concatenate([pose.ravel(), left.ravel(), right.ravel(), face.ravel()]).astype(np.float32)


def find_sequence_dir(keypoint_root: Path, sequence_name: str) -> Optional[Path]:
    m = re.search(r"REAL(\d{2})", sequence_name)
    if m:
        direct = keypoint_root / m.group(1) / sequence_name
        if direct.exists():
            return direct

    for section in sorted([p for p in keypoint_root.iterdir() if p.is_dir()]):
        cand = section / sequence_name
        if cand.exists():
            return cand
    return None


def load_word_mapping(mapping_csv: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[str(row.get("label", "")).strip()] = str(row.get("korean_word", "")).strip()
    return out


def parse_word_code(sequence_name: str) -> Optional[str]:
    m = re.search(r"WORD\d{4}", sequence_name)
    return m.group(0) if m else None


def run_eval(args):
    project_root = Path(__file__).resolve().parent.parent
    service_dir = project_root / "word_Ai+degree_AI" / "services"

    import sys
    if str(service_dir) not in sys.path:
        sys.path.insert(0, str(service_dir))

    from integrated_word_service import IntegratedWordService

    model_base = project_root / "word_AI" / "Final_Model_GRU" / "artifacts" / "Final_GRU_HANDS_126D" / "models"
    model_path = model_base / "best_model.keras"
    label_path = model_base / "label_map.json"
    mapping_path = model_base / "word_label_mapping.csv"
    degree_model_path = project_root / "degree_AI" / "models" / "degree_frame_280d_anger_mlp.joblib"

    service = IntegratedWordService(
        str(model_path),
        str(label_path),
        str(mapping_path),
        str(degree_model_path),
    )
    service.word_engine.threshold = args.threshold

    word_map = load_word_mapping(mapping_path)

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    results = []
    missing_dirs = 0
    no_success = 0

    for i, row in enumerate(rows, start=1):
        seq = str(row.get("sequence_name", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        manual_degree = normalize_degree(row.get("manual_degree"), default="normal")
        word_code = parse_word_code(seq)
        expected_word = word_map.get(word_code or "", "")

        sequence_dir = find_sequence_dir(args.keypoint_root, seq)
        if sequence_dir is None:
            missing_dirs += 1
            results.append({
                "sample_id": sample_id,
                "sequence_name": seq,
                "manual_degree": manual_degree,
                "expected_word": expected_word,
                "status": "missing_sequence_dir",
                "frame_count": 0,
                "pred_word": "",
                "pred_degree_ko": "",
                "pred_degree": "",
                "final_sentence": "",
                "modifier": "",
                "word_match": 0,
                "degree_match": 0,
                "modifier_pass": "FAIL",
                "overall_pass": 0,
                "reason": "키포인트 디렉터리 없음",
            })
            continue

        frame_files = sorted(sequence_dir.glob("*_keypoints.json"))
        # Offline: read all frames, build full-feature list and run a single sequence prediction
        features = []
        for frame_file in frame_files:
            with open(frame_file, "r", encoding="utf-8") as ff:
                frame_data = json.load(ff)
            feature = extract_frame_feature(frame_data)
            features.append(feature)

        # Use offline sequence prediction to match training sampling (uniform resize)
        success_payload = service.process_sequence(features)

        if success_payload is None:
            no_success += 1
            results.append({
                "sample_id": sample_id,
                "sequence_name": seq,
                "manual_degree": manual_degree,
                "expected_word": expected_word,
                "status": "no_success",
                "frame_count": len(frame_files),
                "pred_word": "",
                "pred_degree_ko": "",
                "pred_degree": "",
                "final_sentence": "",
                "modifier": "",
                "word_match": 0,
                "degree_match": 0,
                "modifier_pass": "FAIL",
                "overall_pass": 0,
                "reason": "시퀀스 전체에서 success 미발생",
            })
            continue

        pred_word = str(success_payload.get("original_word", "")).strip()
        pred_degree_ko = str(success_payload.get("predicted_degree", "")).strip()
        pred_degree = normalize_degree(pred_degree_ko, default="normal")
        final_sentence = str(success_payload.get("final_sentence", "")).strip()
        modifier = str(success_payload.get("modifier", "")).strip()
        reason = str(success_payload.get("reason", "")).strip()

        word_match = int(bool(expected_word) and expected_word == pred_word)
        degree_match = int(pred_degree == manual_degree)
        modifier_pass = check_degree_modifier(final_sentence, manual_degree)
        overall_pass = int(word_match == 1 and degree_match == 1 and modifier_pass == "PASS")

        results.append({
            "sample_id": sample_id,
            "sequence_name": seq,
            "manual_degree": manual_degree,
            "expected_word": expected_word,
            "status": "ok",
            "frame_count": len(frame_files),
            "pred_word": pred_word,
            "pred_degree_ko": pred_degree_ko,
            "pred_degree": pred_degree,
            "final_sentence": final_sentence,
            "modifier": modifier,
            "word_match": word_match,
            "degree_match": degree_match,
            "modifier_pass": modifier_pass,
            "overall_pass": overall_pass,
            "reason": reason,
        })

        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] 처리 완료")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.output_dir / "word_level_eval_results.csv"
    fail_csv = args.output_dir / "word_level_eval_fail_cases.csv"
    metrics_json = args.output_dir / "word_level_eval_metrics.json"

    with open(result_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)

    fail_rows = [r for r in results if r.get("overall_pass", 0) == 0]
    with open(fail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(fail_rows)

    total = len(results)
    word_acc = sum(r.get("word_match", 0) for r in results) / total if total else 0.0
    degree_acc = sum(r.get("degree_match", 0) for r in results) / total if total else 0.0
    modifier_acc = sum(1 for r in results if r.get("modifier_pass") == "PASS") / total if total else 0.0
    overall_acc = sum(r.get("overall_pass", 0) for r in results) / total if total else 0.0

    metrics = {
        "total": total,
        "word_accuracy": round(word_acc, 4),
        "degree_accuracy": round(degree_acc, 4),
        "modifier_pass_rate": round(modifier_acc, 4),
        "overall_pass_rate": round(overall_acc, 4),
        "missing_sequence_dir_count": missing_dirs,
        "no_success_count": no_success,
        "threshold": args.threshold,
        "input_csv": str(args.input_csv),
        "keypoint_root": str(args.keypoint_root),
        "result_csv": str(result_csv),
        "fail_csv": str(fail_csv),
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Word-Level Integrated Eval 완료 ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="WORD+degree+LLM 통합 100샘플 평가")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("성능검증용 데이터/degree_manual_full100_eval/combined_manual_degree_100_with_predictions.csv"),
        help="평가 대상 CSV",
    )
    parser.add_argument(
        "--keypoint-root",
        type=Path,
        default=Path(r"C:/Users/USER/Desktop/대학교/26-1학기/SW프로젝트기초/과제/03-1 최종 프로젝트 데이터/real_word 데이터/extracted_keypoints_front"),
        help="실제 키포인트 루트",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/word_level_eval"),
        help="결과 출력 폴더",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="처리 샘플 수 (디폴트 100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="단어 모델 confidence threshold",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
