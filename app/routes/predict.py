import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.degree_service_wrapper import predict_degree
from app.services.semantic_service_wrapper import apply_semantic_postprocess
from app.services.sentence_service_wrapper import predict_sentence
from app.services.video_keypoint_extractor import (
    extract_mediapipe_frame_from_image_bytes,
    summarize_mediapipe_frames,
)
from app.services.word_service_wrapper import predict_word


router = APIRouter(prefix="/api/predict", tags=["predict"])

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_OUTPUT_MODES = {
    "word",
    "sentence",
    "degree",
    "word_degree",
    "sentence_degree",
}
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEGREE_OUTPUT_MODES = {"degree", "word_degree", "sentence_degree"}
UNRECOGNIZED_TEXT = "인식불가"
UNRECOGNIZED_STATUSES = {"low_confidence"}


def _result_failed(result: dict) -> bool:
    model_status = str(result.get("model_status", ""))
    return result.get("status") == "error" or model_status.endswith("_error")


def _raw_result_unrecognized(result: dict) -> bool:
    text = str(result.get("text", "")).strip()
    return text == UNRECOGNIZED_TEXT or result.get("status") in UNRECOGNIZED_STATUSES


def _skipped_degree_result() -> dict:
    return {
        "status": "skipped",
        "degree": "normal",
        "degree_ko": "보통",
        "confidence": 0.0,
        "prob_weak": 0.0,
        "prob_normal": 1.0,
        "prob_strong": 0.0,
        "model_status": "degree_not_requested",
        "message": "현재 출력 모드에서는 표현 정도 모델을 실행하지 않았습니다.",
    }


def _skipped_semantic_result(reason: str) -> dict:
    return {
        "apply_degree": False,
        "final_text": "",
        "target_expression": "",
        "modifier": "",
        "reason": reason,
        "processor_status": "skipped_model_error",
    }


def _unrecognized_semantic_result() -> dict:
    return {
        "apply_degree": False,
        "final_text": UNRECOGNIZED_TEXT,
        "target_expression": "",
        "modifier": "",
        "reason": "원문 인식 결과가 인식불가이므로 표현 정도 후처리를 적용하지 않습니다.",
        "processor_status": "unrecognized",
    }


def _summary_to_keypoints(summary: dict) -> dict:
    return {
        "hands": bool(summary.get("has_left_hand") or summary.get("has_right_hand")),
        "face": bool(summary.get("has_face")),
        "pose": bool(summary.get("has_pose")),
    }


def _save_result(response: dict) -> str:
    result_filename = (
        f"result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.json"
    )
    result_path = RESULT_DIR / result_filename

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=2)

    return str(result_path)


async def _predict_video(
    file: UploadFile,
    mode: str,
    output_mode: str,
    input_type: str,
) -> dict:
    if mode not in {"word", "sentence"}:
        raise HTTPException(status_code=400, detail="mode는 word 또는 sentence만 가능합니다.")

    if output_mode not in ALLOWED_OUTPUT_MODES:
        raise HTTPException(status_code=400, detail="지원하지 않는 출력 모드입니다.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="업로드된 파일이 없습니다.")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {supported}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="파일 내용이 비어 있습니다.")

    saved_filename = f"{uuid.uuid4()}{extension}"
    saved_path = UPLOAD_DIR / saved_filename
    saved_path.write_bytes(content)

    needs_degree = output_mode in DEGREE_OUTPUT_MODES
    degree_result = predict_degree(str(saved_path)) if needs_degree else _skipped_degree_result()

    if output_mode == "degree":
        degree_failed = _result_failed(degree_result)
        raw_ai_result = {
            "text": "" if degree_failed else degree_result["degree_ko"],
            "confidence": degree_result.get("confidence", 0.0),
            "status": "error" if degree_failed else "success",
            "model_status": degree_result.get("model_status", "degree_only"),
            "keypoint_summary": degree_result.get("keypoint_summary", {}),
            "message": degree_result.get("message", "표현 정도 분석 결과입니다."),
        }

        if degree_failed:
            semantic_result = _skipped_semantic_result(
                "표현 정도 모델 오류로 후처리를 실행하지 않았습니다."
            )
            final_result = {"text": "", "modified": False}
        else:
            semantic_result = {
                "apply_degree": False,
                "final_text": f"표현 정도: {degree_result['degree_ko']}",
                "target_expression": "",
                "modifier": "",
                "reason": "표현 정도 단독 출력 모드입니다.",
                "processor_status": "degree_only",
            }
            final_result = {
                "text": semantic_result["final_text"],
                "modified": False,
            }
    else:
        use_sentence = output_mode in {"sentence", "sentence_degree"} or mode == "sentence"
        raw_ai_result = (
            predict_sentence(str(saved_path), source_name=file.filename)
            if use_sentence
            else predict_word(str(saved_path))
        )

        raw_failed = _result_failed(raw_ai_result)
        degree_failed = needs_degree and _result_failed(degree_result)

        if raw_failed or degree_failed:
            semantic_result = _skipped_semantic_result(
                "요청한 모델 중 오류가 발생해 후처리를 실행하지 않았습니다."
            )
            final_result = {"text": "", "modified": False}
        elif _raw_result_unrecognized(raw_ai_result):
            semantic_result = _unrecognized_semantic_result()
            raw_ai_result["text"] = UNRECOGNIZED_TEXT
            final_result = {
                "text": UNRECOGNIZED_TEXT,
                "modified": False,
            }
        else:
            semantic_result = apply_semantic_postprocess(
                mode=mode,
                text=raw_ai_result["text"],
                degree=degree_result["degree"],
                degree_ko=degree_result["degree_ko"],
            )

            if output_mode in {"word", "sentence"}:
                semantic_result["final_text"] = raw_ai_result["text"]
                semantic_result["apply_degree"] = False
                semantic_result["reason"] = (
                    "단독 출력 모드이므로 표현 정도는 실행하지 않습니다."
                )

            final_result = {
                "text": semantic_result["final_text"],
                "modified": semantic_result["apply_degree"],
            }

    requested_results = [raw_ai_result]
    if needs_degree:
        requested_results.append(degree_result)

    failed_results = [result for result in requested_results if _result_failed(result)]
    response_status = "error" if failed_results else "success"
    response_message = (
        failed_results[0].get("message", "모델 분석에 실패했습니다.")
        if failed_results
        else "모델 분석을 완료했습니다."
    )

    response = {
        "status": response_status,
        "message": response_message,
        "mode": mode,
        "output_mode": output_mode,
        "input_type": input_type,
        "file": {
            "original_name": file.filename,
            "saved_name": saved_filename,
            "saved_path": str(saved_path),
        },
        "raw_ai_result": raw_ai_result,
        "degree_result": degree_result,
        "semantic_llm_result": semantic_result,
        "final_result": final_result,
    }
    response["result_path"] = _save_result(response)
    return response


@router.post("/upload")
async def predict_upload(
    file: UploadFile = File(...),
    mode: str = Form("word"),
    output_mode: str = Form("word_degree"),
):
    return await _predict_video(file, mode, output_mode, input_type="upload")


@router.post("/webcam")
async def predict_webcam(
    file: UploadFile = File(...),
    mode: str = Form("word"),
    output_mode: str = Form("word_degree"),
):
    return await _predict_video(file, mode, output_mode, input_type="webcam")


@router.post("/keypoints")
async def detect_keypoints(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="이미지 내용이 비어 있습니다.")

    try:
        frame = extract_mediapipe_frame_from_image_bytes(content)
        summary = summarize_mediapipe_frames([frame])
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "status": "success",
        "keypoints": _summary_to_keypoints(summary),
        "keypoint_summary": summary,
    }
