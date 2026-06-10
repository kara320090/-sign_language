import json
import os
import re
import urllib.error
import urllib.request


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_SENTENCE_MODEL = os.environ.get("OLLAMA_SENTENCE_MODEL", "qwen2.5:3b")
OLLAMA_WORD_MODEL = os.environ.get("OLLAMA_WORD_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "180"))

EMOTION_STATE_TERMS = [
    "화났다",
    "화난다",
    "화나다",
    "슬프다",
    "무섭다",
    "행복하다",
    "기쁘다",
    "아프다",
    "피곤하다",
    "걱정된다",
    "불안하다",
    "좋다",
    "싫다",
    "힘들다",
    "우울하다",
    "짜증난다",
    "답답하다",
]

REQUIRED_KEYS = [
    "apply_degree",
    "final_text",
    "target_expression",
    "modifier",
    "reason",
]


def _modifier_for_degree(degree):
    if degree == "weak":
        return "조금"
    if degree == "strong":
        return "매우"
    return ""


def _ollama_enabled():
    return os.environ.get("OLLAMA_ENABLED", "1").lower() not in {"0", "false", "no"}


def _ollama_model_for_mode(mode: str) -> str:
    if mode == "word":
        return OLLAMA_WORD_MODEL
    return OLLAMA_SENTENCE_MODEL


def _rule_semantic_postprocess(mode: str, text: str, degree: str, degree_ko: str) -> dict:
    modifier = _modifier_for_degree(str(degree).lower())

    if not text or not modifier:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "텍스트가 비어 있거나 표현 정도가 보통이므로 원문을 유지합니다.",
            "processor_status": "rule",
        }

    if mode == "sentence":
        for term in sorted(EMOTION_STATE_TERMS, key=len, reverse=True):
            if term in text:
                return {
                    "apply_degree": True,
                    "final_text": text.replace(term, f"{modifier} {term}", 1),
                    "target_expression": term,
                    "modifier": modifier,
                    "reason": f"문장 내 감정/상태 표현에 {degree_ko} 정도를 반영했습니다.",
                    "processor_status": "rule_guard",
                }

        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "일반 문장으로 판단해 원문을 유지합니다.",
            "processor_status": "rule_guard",
        }

    return {
        "apply_degree": True,
        "final_text": f"{modifier} {text}",
        "target_expression": text,
        "modifier": modifier,
        "reason": f"단어 결과에 {degree_ko} 정도를 반영했습니다.",
        "processor_status": "rule",
    }


def _build_ollama_prompt(
    mode: str,
    text: str,
    degree: str,
    degree_ko: str,
    candidate_text: str,
) -> str:
    return f"""
너는 수어 인식 결과를 자연스러운 한국어로 후처리하는 엔진이다.
반드시 JSON 객체 하나만 출력한다.

입력:
- mode: {mode}
- 원문: {text}
- 표현 정도: {degree}
- 표현 정도 한글: {degree_ko}
- 안전 후보: {candidate_text}

규칙:
1. 원문의 의미를 바꾸거나 새 정보를 추가하지 않는다.
2. degree가 normal이면 final_text는 원문과 완전히 같아야 한다.
3. degree가 weak이면 필요한 경우 '조금'을 넣는다.
4. degree가 strong이면 필요한 경우 '매우'를 넣는다.
5. mode가 word이면 수식어와 원문 단어를 결합한다.
6. mode가 sentence이면 감정이나 상태 표현에만 정도를 반영한다.
7. 일반 행동 문장이라 정도 표현이 어색하면 원문을 유지하고 apply_degree는 false로 둔다.
8. modifier는 final_text에 실제로 들어간 수식어와 정확히 같아야 한다.
9. final_text는 원문 전체 문장을 보존해야 하며 주어, 목적어, 조사, 어미를 삭제하면 안 된다.
10. 안전 후보가 자연스러우면 안전 후보를 그대로 final_text로 사용한다.

예시:
- 원문 "나는 화났다", strong, 안전 후보 "나는 매우 화났다"이면 final_text는 "나는 매우 화났다"이다.
- 원문 "나는 학교에 간다", strong, 안전 후보 "나는 학교에 간다"이면 final_text는 "나는 학교에 간다"이고 apply_degree는 false이다.
- 원문 "사과", strong, 안전 후보 "매우 사과"이면 final_text는 "매우 사과"이다.

출력 형식:
{{
  "apply_degree": true,
  "final_text": "최종 문장",
  "target_expression": "정도를 반영한 원문 표현",
  "modifier": "조금 또는 매우 또는 빈 문자열",
  "reason": "처리 이유"
}}
""".strip()


def _extract_json(response_text: str) -> dict:
    response_text = str(response_text or "").strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError("Ollama 응답에서 JSON 객체를 찾을 수 없습니다.")

    return json.loads(match.group(0))


def _normalize_for_compare(value: str) -> str:
    value = re.sub(r"[\s\"'`.,!?;:()\[\]{}]+", "", str(value or ""))
    return value


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _call_ollama(mode: str, prompt: str) -> str:
    payload = {
        "model": _ollama_model_for_mode(mode),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SEC) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error


def _validate_ollama_result(result: dict, mode: str, text: str, degree: str) -> dict:
    for key in REQUIRED_KEYS:
        if key not in result:
            raise ValueError(f"Ollama 응답에 {key} 필드가 없습니다.")

    final_text = str(result.get("final_text", "")).strip()
    target_expression = str(result.get("target_expression", "")).strip()
    modifier = str(result.get("modifier", "")).strip()
    reason = str(result.get("reason", "")).strip()
    apply_degree = _parse_bool(result.get("apply_degree"))
    expected_modifier = _modifier_for_degree(degree)

    if not final_text:
        raise ValueError("Ollama final_text가 비어 있습니다.")

    if len(final_text) > max(len(text) + 40, 80):
        raise ValueError("Ollama final_text가 원문보다 지나치게 깁니다.")

    if not expected_modifier:
        if final_text != text:
            raise ValueError("normal 정도에서 Ollama가 원문을 변경했습니다.")
    elif apply_degree:
        if modifier != expected_modifier and expected_modifier not in final_text:
            raise ValueError("Ollama modifier가 표현 정도와 일치하지 않습니다.")
        restored = final_text.replace(expected_modifier, "", 1).strip()
        if _normalize_for_compare(restored) != _normalize_for_compare(text):
            raise ValueError("Ollama 결과가 원문 구조를 벗어났습니다.")
        if mode == "sentence" and target_expression and target_expression not in text:
            target_expression = text
        modifier = expected_modifier
    elif _normalize_for_compare(final_text) != _normalize_for_compare(text):
        raise ValueError("apply_degree가 false인데 Ollama가 원문을 변경했습니다.")

    return {
        "apply_degree": apply_degree,
        "final_text": final_text,
        "target_expression": target_expression,
        "modifier": modifier,
        "reason": reason or "Ollama 후처리를 적용했습니다.",
        "processor_status": "ollama",
        "ollama_model": _ollama_model_for_mode(mode),
    }


def _ollama_semantic_postprocess(
    mode: str,
    text: str,
    degree: str,
    degree_ko: str,
    rule_result: dict,
) -> dict:
    prompt = _build_ollama_prompt(
        mode,
        text,
        degree,
        degree_ko,
        str(rule_result.get("final_text", text)),
    )
    raw_response = _call_ollama(mode, prompt)
    parsed = _extract_json(raw_response)
    return _validate_ollama_result(parsed, mode, text, degree)


def _ollama_fallback(rule_result: dict, error: Exception) -> dict:
    result = dict(rule_result)
    result["processor_status"] = "ollama_fallback"
    result["fallback_processor_status"] = rule_result.get("processor_status", "rule")
    result["ollama_error"] = str(error)
    result["reason"] = f"Ollama 호출 실패로 규칙 기반 후처리를 사용했습니다. {rule_result.get('reason', '')}".strip()
    return result


def apply_semantic_postprocess(mode: str, text: str, degree: str, degree_ko: str) -> dict:
    text = str(text or "").strip()
    degree = str(degree or "").strip().lower()
    degree_ko = str(degree_ko or "").strip()
    rule_result = _rule_semantic_postprocess(mode, text, degree, degree_ko)

    if not _ollama_enabled() or not text or degree == "normal":
        return rule_result

    try:
        return _ollama_semantic_postprocess(mode, text, degree, degree_ko, rule_result)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        return _ollama_fallback(rule_result, error)
    except Exception as error:
        return _ollama_fallback(rule_result, error)
