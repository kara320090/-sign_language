from pathlib import Path

import joblib
import numpy as np

from app.services.video_keypoint_extractor import (
    extract_mediapipe_frames_from_video,
    preprocess_degree_mediapipe_frames,
    summarize_mediapipe_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEGREE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "모델" / "정도표현" / "degree_frame_280d_anger_mlp.joblib",
    PROJECT_ROOT / "degree_AI" / "models" / "degree_frame_280d_anger_mlp.joblib",
]

DEFAULT_LABEL_NAMES = ["weak", "normal", "strong"]
DEGREE_KO = {
    "weak": "약함",
    "normal": "보통",
    "strong": "강함",
}

_degree_bundle = None
_degree_model_path = None


def _find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def get_degree_bundle():
    global _degree_bundle, _degree_model_path

    if _degree_bundle is not None:
        return _degree_bundle

    model_path = _find_existing_path(DEGREE_MODEL_CANDIDATES)
    if model_path is None:
        searched = "\n".join(str(path) for path in DEGREE_MODEL_CANDIDATES)
        raise FileNotFoundError(f"degree_AI 모델 파일을 찾을 수 없습니다.\n{searched}")

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        bundle = loaded
    else:
        bundle = {
            "model": loaded,
            "label_names": DEFAULT_LABEL_NAMES,
            "input_shape": [280],
        }

    model = bundle["model"]
    if not hasattr(model, "predict"):
        raise TypeError(f"degree 모델에 predict 함수가 없습니다: {type(model).__name__}")

    expected_features = int(getattr(model, "n_features_in_", 280))
    if expected_features != 280:
        raise ValueError(f"degree 모델 입력 차원이 280D가 아닙니다: {expected_features}")

    _degree_bundle = bundle
    _degree_model_path = model_path
    return _degree_bundle


def _predict_average_probabilities(model, features, label_count):
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float32)
        return np.mean(probabilities, axis=0)

    predictions = np.asarray(model.predict(features), dtype=np.int64)
    probabilities = np.zeros((len(predictions), label_count), dtype=np.float32)

    for row, prediction in enumerate(predictions):
        if 0 <= int(prediction) < label_count:
            probabilities[row, int(prediction)] = 1.0

    return np.mean(probabilities, axis=0)


def predict_degree(video_path: str) -> dict:
    try:
        frames = extract_mediapipe_frames_from_video(video_path)
        keypoint_summary = summarize_mediapipe_frames(frames)
        features, has_face = preprocess_degree_mediapipe_frames(frames)

        if not has_face:
            return {
                "status": "success",
                "degree": "normal",
                "degree_ko": "보통",
                "confidence": 0.0,
                "prob_weak": 0.0,
                "prob_normal": 1.0,
                "prob_strong": 0.0,
                "has_face": False,
                "num_used_frames": 0,
                "model_status": "degree_ai_no_face",
                "keypoint_extractor": "mediapipe",
                "keypoint_summary": keypoint_summary,
                "model_input_shape": list(features.shape),
                "message": "얼굴 keypoint가 검출되지 않아 표현 정도를 보통으로 반환했습니다.",
            }

        try:
            bundle = get_degree_bundle()
            model = bundle["model"]
            label_names = list(bundle.get("label_names", DEFAULT_LABEL_NAMES))
            probabilities = _predict_average_probabilities(model, features, len(label_names))
            best_index = int(np.argmax(probabilities))
            degree = label_names[best_index]

            probability_by_label = {
                label: float(probabilities[index])
                for index, label in enumerate(label_names)
            }

            return {
                "status": "success",
                "degree": degree,
                "degree_ko": DEGREE_KO.get(degree, degree),
                "confidence": float(probabilities[best_index]),
                "prob_weak": probability_by_label.get("weak", 0.0),
                "prob_normal": probability_by_label.get("normal", 0.0),
                "prob_strong": probability_by_label.get("strong", 0.0),
                "has_face": has_face,
                "num_used_frames": int(len(features)),
                "model_status": "degree_ai_frame_280d_mlp_connected",
                "model_path": str(_degree_model_path),
                "keypoint_extractor": "mediapipe",
                "keypoint_summary": keypoint_summary,
                "model_input_shape": list(features.shape),
                "model_input_type": "valid MediaPipe face frames, averaged 1F×280D predictions",
                "message": "Valid MediaPipe face frames were preprocessed with the degree-training pipeline before MLP inference.",
            }

        except Exception as model_error:
            return {
                "status": "error",
                "degree": "normal",
                "degree_ko": "보통",
                "confidence": 0.0,
                "prob_weak": 0.0,
                "prob_normal": 1.0,
                "prob_strong": 0.0,
                "has_face": has_face,
                "model_status": "degree_ai_model_error",
                "keypoint_extractor": "mediapipe",
                "keypoint_summary": keypoint_summary,
                "model_input_shape": list(features.shape),
                "message": f"degree 모델 연결 또는 추론에 실패했습니다: {model_error}",
            }

    except Exception as error:
        return {
            "status": "error",
            "degree": "normal",
            "degree_ko": "보통",
            "confidence": 0.0,
            "prob_weak": 0.0,
            "prob_normal": 1.0,
            "prob_strong": 0.0,
            "model_status": "degree_ai_error",
            "message": str(error),
        }
