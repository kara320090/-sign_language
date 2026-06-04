import os
import uuid
import json
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.word_service_wrapper import predict_word
from app.services.sentence_service_wrapper import predict_sentence
from app.services.degree_service_wrapper import predict_degree
from app.services.semantic_service_wrapper import apply_semantic_postprocess

router = APIRouter(prefix="/api/predict", tags=["predict"])

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@router.post("/upload")
async def predict_upload(
    file: UploadFile = File(...),
    mode: str = Form("word"),
    output_mode: str = Form("word_degree")
):
    try:
        if mode not in ["word", "sentence"]:
            raise HTTPException(
                status_code=400,
                detail="mode는 word 또는 sentence만 가능합니다."
            )

        allowed_output_modes = [
            "word",
            "sentence",
            "degree",
            "word_degree",
            "sentence_degree"
        ]

        if output_mode not in allowed_output_modes:
            output_mode = "word_degree"

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="업로드된 파일이 없습니다."
            )

        allowed_ext = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 형식입니다. mp4, avi, mov, mkv 파일만 업로드할 수 있습니다."
            )

        saved_filename = f"{uuid.uuid4()}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)

        content = await file.read()

        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="파일 내용이 비어 있습니다."
            )

        with open(saved_path, "wb") as f:
            f.write(content)

        # 1. 모든 모드에서 degree는 계산한다.
        degree_result = predict_degree(saved_path)

        # 2. 출력 모드에 따라 raw_ai_result 구성
        if output_mode == "degree":
            raw_ai_result = {
                "text": degree_result["degree_ko"],
                "confidence": degree_result["confidence"],
                "status": "success",
                "model_status": degree_result.get("model_status", "degree_only"),
                "message": "표현정도만 분석한 결과입니다."
            }

            semantic_llm_result = {
                "apply_degree": False,
                "final_text": f"표현 정도: {degree_result['degree_ko']}",
                "target_expression": "",
                "modifier": "",
                "reason": "표현정도 단독 출력 모드입니다.",
                "processor_status": "degree_only"
            }

            final_result = {
                "text": semantic_llm_result["final_text"],
                "modified": False
            }

        else:
            if output_mode in ["sentence", "sentence_degree"] or mode == "sentence":
                raw_ai_result = predict_sentence(saved_path)
            else:
                raw_ai_result = predict_word(saved_path)

            semantic_llm_result = apply_semantic_postprocess(
                mode=mode,
                text=raw_ai_result["text"],
                degree=degree_result["degree"],
                degree_ko=degree_result["degree_ko"]
            )

            # 단어만 / 문장만 모드에서는 degree를 최종 문장에 강제로 반영하지 않음
            if output_mode in ["word", "sentence"]:
                final_result = {
                    "text": raw_ai_result["text"],
                    "modified": False
                }
                semantic_llm_result["final_text"] = raw_ai_result["text"]
                semantic_llm_result["apply_degree"] = False
                semantic_llm_result["reason"] = "단독 출력 모드이므로 표현정도는 별도 정보로만 표시합니다."

                final_result = {
                    "text": raw_ai_result["text"],
                    "modified": False
        }
            else:
                final_result = {
                    "text": semantic_llm_result["final_text"],
                    "modified": semantic_llm_result["apply_degree"]
                }

        response = {
            "status": "success",
            "mode": mode,
            "output_mode": output_mode,
            "input_type": "upload",
            "file": {
                "original_name": file.filename,
                "saved_name": saved_filename,
                "saved_path": saved_path
            },
            "raw_ai_result": raw_ai_result,
            "degree_result": degree_result,
            "semantic_llm_result": semantic_llm_result,
            "final_result": final_result
        }

        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(RESULT_DIR, result_filename)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        response["result_path"] = result_path

        return response

    except HTTPException:
        raise

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }