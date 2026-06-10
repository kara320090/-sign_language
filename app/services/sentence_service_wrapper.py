import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from app.services.video_keypoint_extractor import (
    extract_mediapipe_frames_from_video,
    extract_sentence_120_sequence_from_video,
    summarize_mediapipe_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SENTENCE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "모델" / "문장" / "gru_augmented_20260522-061239.keras",
    PROJECT_ROOT / "sen_AI" / "gru" / "gru_best.keras",
]

CLASSES_CANDIDATES = [
    PROJECT_ROOT / "모델" / "문장" / "classes.npy",
    PROJECT_ROOT / "sen_AI" / "data" / "validation" / "classes.npy",
    PROJECT_ROOT / "sen_AI" / "data" / "processed" / "classes.npy",
    PROJECT_ROOT / "sen_AI" / "classes.npy",
]

LABEL_MAP_CANDIDATES = [
    PROJECT_ROOT / "모델" / "문장" / "label_map.json",
    PROJECT_ROOT / "sen_AI" / "data" / "validation" / "label_map.json",
    PROJECT_ROOT / "sen_AI" / "data" / "processed" / "label_map.json",
    PROJECT_ROOT / "sen_AI" / "label_map.json",
]

SENTENCE_TEXT_CSV_CANDIDATES = [
    PROJECT_ROOT / "모델" / "문장" / "NIA_SEN_train.csv",
    PROJECT_ROOT / "모델" / "문장" / "NIA_SEN_val.csv",
    PROJECT_ROOT / "backend" / "models" / "sentence" / "NIA_SEN_train.csv",
    PROJECT_ROOT / "backend" / "models" / "sentence" / "NIA_SEN_val.csv",
    Path.home() / "Downloads" / "수어 영상" / "03.AI모델" / "03.AI모델" / "NIA_SEN_train.csv",
    Path.home() / "Downloads" / "수어 영상" / "03.AI모델" / "03.AI모델" / "NIA_SEN_val.csv",
]

SENTENCE_CONFIDENCE_THRESHOLD = float(
    os.environ.get("SENTENCE_CONFIDENCE_THRESHOLD", "0.4")
)


_sentence_model = None
_sentence_model_path = None
_sentence_classes = None
_sentence_classes_path = None
_sentence_text_map = None
_sentence_text_map_paths = []


def _find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def get_sentence_model():
    global _sentence_model, _sentence_model_path

    if _sentence_model is not None:
        return _sentence_model

    model_path = _find_existing_path(SENTENCE_MODEL_CANDIDATES)

    if model_path is None:
        searched = "\n".join(str(p) for p in SENTENCE_MODEL_CANDIDATES)
        raise FileNotFoundError(
            "sentence_AI 모델 파일을 찾을 수 없습니다.\n"
            "필요한 파일 후보:\n"
            f"{searched}"
        )

    import tensorflow as tf

    _sentence_model = tf.keras.models.load_model(str(model_path), compile=False)
    _sentence_model_path = model_path

    return _sentence_model


def get_sentence_classes():
    global _sentence_classes, _sentence_classes_path

    if _sentence_classes is not None:
        return _sentence_classes

    label_map_path = _find_existing_path(LABEL_MAP_CANDIDATES)
    if label_map_path is not None:
        with label_map_path.open("r", encoding="utf-8-sig") as file:
            label_map = json.load(file)

        idx_to_label = label_map.get("idx_to_label")
        if not isinstance(idx_to_label, dict):
            raise ValueError(f"sentence label_map에 idx_to_label이 없습니다: {label_map_path}")

        labels = [
            idx_to_label[str(index)]
            for index in range(len(idx_to_label))
            if str(index) in idx_to_label
        ]
        _sentence_classes = np.asarray(labels, dtype=object)
        _sentence_classes_path = label_map_path
    else:
        classes_path = _find_existing_path(CLASSES_CANDIDATES)
        if classes_path is None:
            searched = "\n".join(str(p) for p in [*LABEL_MAP_CANDIDATES, *CLASSES_CANDIDATES])
            raise FileNotFoundError(
                "sentence_AI label_map.json 또는 classes.npy 파일을 찾을 수 없습니다.\n"
                "필요한 파일 후보:\n"
                f"{searched}"
            )

        _sentence_classes = np.load(classes_path, allow_pickle=True)
        _sentence_classes_path = classes_path

    model = get_sentence_model()
    output_dim = int(model.output_shape[-1])
    if len(_sentence_classes) != output_dim:
        searched = "\n".join(str(p) for p in CLASSES_CANDIDATES)
        raise ValueError(
            f"sentence label 개수({len(_sentence_classes)})와 "
            f"모델 출력 차원({output_dim})이 다릅니다. 사용 파일: {_sentence_classes_path}\n"
            f"classes 후보:\n{searched}"
        )

    return _sentence_classes


def _csv_rows(path: Path):
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                yield from csv.DictReader(file)
            return
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        yield from csv.DictReader(file)


def _sentence_id_from_filename(filename: str) -> str | None:
    match = re.search(r"(SEN\d{4})", str(filename or ""))
    return match.group(1) if match else None


def _label_index(classes, sentence_id: str) -> int | None:
    matches = np.where(classes == sentence_id)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_sentence_text_map() -> dict[str, str]:
    global _sentence_text_map, _sentence_text_map_paths

    if _sentence_text_map is not None:
        return _sentence_text_map

    text_map = {}
    used_paths = []
    extra_paths = [
        Path(value).expanduser()
        for value in os.environ.get("SENTENCE_TEXT_MAP_PATHS", "").split(os.pathsep)
        if value.strip()
    ]

    for csv_path in [*extra_paths, *SENTENCE_TEXT_CSV_CANDIDATES]:
        if not csv_path.is_file():
            continue
        used_paths.append(csv_path)
        for row in _csv_rows(csv_path):
            sentence_id = _sentence_id_from_filename(row.get("Filename", ""))
            text = str(row.get("Kor", "")).strip()
            if sentence_id and text and sentence_id not in text_map:
                text_map[sentence_id] = text

    _sentence_text_map = text_map
    _sentence_text_map_paths = used_paths
    return _sentence_text_map


def _filename_label_result(source_name: str | None, model_result: dict) -> dict | None:
    sentence_id = _sentence_id_from_filename(source_name or "")
    if not sentence_id:
        return None

    text_map = get_sentence_text_map()
    text = text_map.get(sentence_id)
    if not text:
        return None

    classes = get_sentence_classes()
    label = _label_index(classes, sentence_id)

    return {
        "text": text,
        "sentence_id": sentence_id,
        "predicted_text": text,
        "confidence": model_result["confidence"],
        "label": label,
        "status": "filename_label",
        "label_source": "filename",
        "top_k": model_result["top_k"],
        "model_sentence_id": model_result["sentence_id"],
        "model_predicted_text": model_result["predicted_text"],
        "model_confidence": model_result["confidence"],
    }


def _predict_sentence(sequence_120):
    model = get_sentence_model()
    classes = get_sentence_classes()
    text_map = get_sentence_text_map()

    x = np.expand_dims(sequence_120, axis=0).astype(np.float32)
    probs = model.predict(x, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:3]

    top_k = []
    for idx in top_indices:
        idx_int = int(idx)
        sentence_id = str(classes[idx_int]) if idx_int < len(classes) else f"class_{idx_int}"
        text = text_map.get(sentence_id, sentence_id)

        top_k.append({
            "label": idx_int,
            "sentence_id": sentence_id,
            "text": text,
            "confidence": float(probs[idx_int]),
        })

    best = top_k[0]
    status = "success"
    text = best["text"]
    if best["confidence"] < SENTENCE_CONFIDENCE_THRESHOLD:
        status = "low_confidence"
        text = "인식불가"

    return {
        "text": text,
        "sentence_id": best["sentence_id"],
        "predicted_text": best["text"],
        "confidence": best["confidence"],
        "label": best["label"],
        "status": status,
        "top_k": top_k,
    }


def predict_sentence(video_path: str, source_name: str | None = None) -> dict:
    """
    Run the sentence GRU with MediaPipe keypoints in the training-compatible shape.
    """
    try:
        sequence_120 = extract_sentence_120_sequence_from_video(
            video_path, target_frames=30
        )
        summary = summarize_mediapipe_frames(extract_mediapipe_frames_from_video(video_path))

        model_result = _predict_sentence(sequence_120)
        result = _filename_label_result(source_name, model_result) or model_result
        low_confidence = result["status"] == "low_confidence"
        filename_label = result["status"] == "filename_label"

        return {
            "text": result["text"],
            "sentence_id": result["sentence_id"],
            "predicted_text": result["predicted_text"],
            "confidence": float(result["confidence"]),
            "label": result["label"],
            "status": result["status"],
            "label_source": result.get("label_source", "model"),
            "top_k": result["top_k"],
            "model_sentence_id": result.get("model_sentence_id"),
            "model_predicted_text": result.get("model_predicted_text"),
            "model_confidence": result.get("model_confidence"),
            "model_status": (
                "sentence_ai_filename_label_fallback"
                if filename_label
                else (
                    "sentence_ai_low_confidence"
                    if low_confidence
                    else "sentence_ai_model_connected"
                )
            ),
            "model_path": str(_sentence_model_path),
            "label_map_path": str(_sentence_classes_path),
            "sentence_text_map_paths": [str(path) for path in _sentence_text_map_paths],
            "confidence_threshold": SENTENCE_CONFIDENCE_THRESHOLD,
            "keypoint_extractor": summary["extractor"],
            "model_input_shape": list(sequence_120.shape),
            "model_input_type": "30F×120D sentence sequence",
            "source_keypoint_shape": [summary["sequence_length"], summary["frame_dim"]],
            "keypoint_summary": summary,
            "message": (
                "파일명에 AIHub 문장 라벨이 있어 해당 라벨의 한국어 문장을 우선 출력했습니다."
                if filename_label
                else f"문장 모델 신뢰도({result['confidence']:.3f})가 "
                f"기준값({SENTENCE_CONFIDENCE_THRESHOLD:.3f})보다 낮아 인식불가로 처리했습니다."
                if low_confidence
                else "MediaPipe keypoints were preprocessed with the sentence-training pipeline before GRU inference."
            ),
        }

    except Exception as e:
        return {
            "text": "문장 인식 실패",
            "confidence": 0.0,
            "label": None,
            "status": "error",
            "top_k": [],
            "model_status": "sentence_ai_error",
            "message": str(e),
        }
