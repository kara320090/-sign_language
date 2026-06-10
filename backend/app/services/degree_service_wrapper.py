from pathlib import Path

import joblib
import numpy as np

from app.services.video_keypoint_extractor import (
    extract_411d_sequence_from_video,
    extract_degree_280_from_411d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEGREE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "degree_AI" / "models" / "degree_frame_280d_anger_mlp.joblib",
    PROJECT_ROOT / "degree_AI" / "models" / "degree_1f_280d_mlp.joblib",
    PROJECT_ROOT / "degree_AI" / "models" / "degree_mlp_280d.joblib",
    PROJECT_ROOT / "degree_AI" / "models" / "best_degree_mlp.joblib",
]

LABEL_CANDIDATES = [
    PROJECT_ROOT / "degree_AI" / "models" / "label_info_seq_anger.joblib",
    PROJECT_ROOT / "degree_AI" / "models" / "label_info_anger.joblib",
    PROJECT_ROOT / "degree_AI" / "models" / "label_info.joblib",
]


_degree_model = None
_degree_model_path = None


def _find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def get_degree_model():
    global _degree_model, _degree_model_path

    if _degree_model is not None:
        return _degree_model

    model_path = _find_existing_path(DEGREE_MODEL_CANDIDATES)

    if model_path is None:
        searched = "\n".join(str(p) for p in DEGREE_MODEL_CANDIDATES)
        raise FileNotFoundError(
            "degree_AI 1F×280D MLP 모델 파일을 찾을 수 없습니다.\n"
            "필요한 파일 후보:\n"
            f"{searched}"
        )

    loaded = joblib.load(model_path)

    # joblib 파일이 dict 형태로 저장된 경우 내부 모델을 꺼낸다.
    if isinstance(loaded, dict):
        for key in ["model", "clf", "classifier", "mlp", "best_model", "estimator"]:
            if key in loaded and hasattr(loaded[key], "predict"):
                loaded = loaded[key]
                break

    if not hasattr(loaded, "predict"):
        raise TypeError(
            f"degree joblib에서 predict 가능한 모델을 찾지 못했습니다. "
            f"로드된 타입: {type(loaded)}"
        )

    _degree_model = loaded
    _degree_model_path = model_path

    return _degree_model


def _normalize_degree_label(label):
    label_str = str(label).lower()

    if label_str in ["0", "weak", "약함"]:
        return "weak", "약함"
    if label_str in ["1", "normal", "보통"]:
        return "normal", "보통"
    if label_str in ["2", "strong", "강함"]:
        return "strong", "강함"

    return "normal", "보통"


def _fallback_rule_degree(feature_280):
    """
    degree_AI 모델 파일이 없을 때만 사용하는 fallback.
    보고서 기준 최종 구조와 구분하기 위해 model_status에 fallback 표시.
    """
    delta = feature_280[16 + 132:]
    score = float(np.mean(np.abs(delta)))

    if score < 0.02:
        degree = "weak"
        degree_ko = "약함"
        probs = [0.72, 0.22, 0.06]
    elif score < 0.06:
        degree = "normal"
        degree_ko = "보통"
        probs = [0.18, 0.68, 0.14]
    else:
        degree = "strong"
        degree_ko = "강함"
        probs = [0.08, 0.22, 0.70]

    return degree, degree_ko, probs, score


def predict_degree(video_path: str) -> dict:
    """
    보고서 기준 degree_AI 구조.

    mp4
    → 30F×411D 공통 keypoint
    → 1F×280D face feature
    → degree_AI MLP 모델
    → weak / normal / strong 반환
    """
    try:
        sequence_411d = extract_411d_sequence_from_video(video_path, target_frames=30)
        feature_280, has_face = extract_degree_280_from_411d(sequence_411d)

        try:
            model = get_degree_model()

            x = feature_280.reshape(1, -1).astype(np.float32)

            pred = model.predict(x)[0]
            degree, degree_ko = _normalize_degree_label(pred)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0]
                proba = np.asarray(proba, dtype=np.float32)

                if len(proba) >= 3:
                    prob_weak = float(proba[0])
                    prob_normal = float(proba[1])
                    prob_strong = float(proba[2])
                else:
                    prob_weak, prob_normal, prob_strong = 0.0, 1.0, 0.0
            else:
                prob_weak = 1.0 if degree == "weak" else 0.0
                prob_normal = 1.0 if degree == "normal" else 0.0
                prob_strong = 1.0 if degree == "strong" else 0.0

            confidence = max(prob_weak, prob_normal, prob_strong)

            return {
                "degree": degree,
                "degree_ko": degree_ko,
                "confidence": float(confidence),
                "prob_weak": float(prob_weak),
                "prob_normal": float(prob_normal),
                "prob_strong": float(prob_strong),
                "has_face": has_face,
                "model_status": "degree_ai_1f_280d_mlp_connected",
                "model_path": str(_degree_model_path),
                "model_input_shape": list(feature_280.shape),
                "model_input_type": "1F×280D face feature",
                "message": "영상에서 1F×280D 얼굴표현 feature를 생성한 뒤 degree_AI MLP 모델 추론을 수행했습니다.",
            }

        except Exception as model_error:
            degree, degree_ko, probs, score = _fallback_rule_degree(feature_280)

            return {
                "degree": degree,
                "degree_ko": degree_ko,
                "confidence": float(max(probs)),
                "prob_weak": float(probs[0]),
                "prob_normal": float(probs[1]),
                "prob_strong": float(probs[2]),
                "score": float(score),
                "has_face": has_face,
                "model_status": "degree_ai_1f_280d_fallback",
                "model_input_shape": list(feature_280.shape),
                "model_input_type": "1F×280D face feature",
                "message": f"1F×280D feature 생성은 성공했지만 degree_AI 모델 연결에 실패하여 fallback을 사용했습니다: {model_error}",
            }

    except Exception as e:
        return {
            "degree": "normal",
            "degree_ko": "보통",
            "confidence": 0.0,
            "prob_weak": 0.0,
            "prob_normal": 1.0,
            "prob_strong": 0.0,
            "model_status": "degree_ai_error",
            "message": str(e),
        }