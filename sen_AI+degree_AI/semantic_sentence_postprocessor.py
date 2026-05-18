import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "sentence_degree_prompt.txt"


EMOTION_STATE_TERMS = [
    "화났다", "화나다",
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
    "놀랐다",
]


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(text: str, degree: str, degree_ko: str) -> str:
    template = load_prompt_template()
    return (
        template
        .replace("{text}", text)
        .replace("{degree}", degree)
        .replace("{degree_ko}", degree_ko)
    )


def extract_json(response_text: str) -> Dict[str, Any]:
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError("JSON object not found in LLM response")

    return json.loads(match.group(0))


def call_ollama(prompt: str, timeout_sec: int = 30) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result.get("response", "")


def fallback_sentence_degree(text: str, degree: str, reason: str) -> Dict[str, Any]:
    return {
        "apply_degree": False,
        "final_text": text,
        "target_expression": "",
        "modifier": "",
        "reason": reason,
        "source": "fallback"
    }


def safe_rule_sentence_degree(text: str, degree: str, reason: str) -> Dict[str, Any]:
    """
    LLM이 원문을 바꾸거나 일반 행동문을 잘못 수정할 경우 사용하는 안전 규칙.
    원문 의미를 바꾸지 않고, 감정/상태 표현에만 '조금/매우'를 삽입한다.
    """
    if degree == "strong":
        modifier = "매우"
    elif degree == "weak":
        modifier = "조금"
    else:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "degree가 normal이므로 원문 유지",
            "source": "rule_guard",
        }

    # 긴 표현부터 먼저 처리
    for term in sorted(EMOTION_STATE_TERMS, key=len, reverse=True):
        if term in text:
            final_text = text.replace(term, f"{modifier} {term}", 1)
            return {
                "apply_degree": True,
                "final_text": final_text,
                "target_expression": term,
                "modifier": modifier,
                "reason": reason,
                "source": "rule_guard",
            }

    return {
        "apply_degree": False,
        "final_text": text,
        "target_expression": "",
        "modifier": "",
        "reason": "감정/상태 표현이 아니므로 원문 유지",
        "source": "rule_guard",
    }


def is_safe_llm_result(result: Dict[str, Any], original_text: str, degree: str) -> bool:
    """
    LLM 결과가 원문을 훼손하지 않았는지 검사한다.
    허용되는 수정은 원문 안의 감정/상태 표현 앞에 '조금/매우'를 삽입하는 것뿐이다.
    """
    final_text = str(result.get("final_text", "")).strip()
    apply_degree = result.get("apply_degree", False)
    target_expression = str(result.get("target_expression", "")).strip()
    modifier = str(result.get("modifier", "")).strip()

    if not final_text:
        return False

    if degree == "normal":
        return final_text == original_text

    if not apply_degree:
        return final_text == original_text

    expected_modifier = "매우" if degree == "strong" else "조금"

    if modifier != expected_modifier:
        return False

    if target_expression not in original_text:
        return False

    if target_expression not in EMOTION_STATE_TERMS:
        return False

    # final_text에서 modifier 하나를 제거했을 때 원문과 같아야 함
    restored = final_text.replace(f"{expected_modifier} ", "", 1).strip()

    if restored != original_text:
        return False

    return True


def validate_llm_result(result: Dict[str, Any], original_text: str, degree: str) -> Dict[str, Any]:
    required_keys = [
        "apply_degree",
        "final_text",
        "target_expression",
        "modifier",
        "reason"
    ]

    for key in required_keys:
        if key not in result:
            raise ValueError(f"missing key: {key}")

    if not isinstance(result["apply_degree"], bool):
        raise ValueError("apply_degree must be boolean")

    if not str(result["final_text"]).strip():
        raise ValueError("final_text is empty")

    if degree == "normal" and result["final_text"] != original_text:
        raise ValueError("degree is normal but final_text was modified")

    return result


def semantic_sentence_postprocess(
    sentence_result: Dict[str, Any],
    degree_result: Dict[str, Any],
    use_llm: bool = True
) -> Dict[str, Any]:
    text = str(sentence_result.get("text", "")).strip()
    degree = str(degree_result.get("degree", "normal")).strip().lower()
    degree_ko = str(degree_result.get("degree_ko", "보통")).strip()

    if not text:
        return fallback_sentence_degree("", degree, "empty sentence text")

    if degree not in {"weak", "normal", "strong"}:
        return fallback_sentence_degree(text, degree, "invalid degree value")

    if degree == "normal":
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "degree가 normal이므로 원문 유지",
            "source": "rule"
        }

    if not use_llm:
        return safe_rule_sentence_degree(
            text=text,
            degree=degree,
            reason="LLM disabled, rule based processing"
        )

    start_time = time.time()

    try:
        prompt = build_prompt(text=text, degree=degree, degree_ko=degree_ko)
        raw_response = call_ollama(prompt)
        parsed = extract_json(raw_response)
        validated = validate_llm_result(parsed, text, degree)

        if not is_safe_llm_result(validated, text, degree):
            guarded = safe_rule_sentence_degree(
                text=text,
                degree=degree,
                reason="LLM 결과가 원문을 변경하거나 일반 행동문을 수정하여 안전 규칙으로 보정함"
            )
            guarded["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return guarded

        validated["source"] = "ollama"
        validated["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return validated

    except (urllib.error.URLError, TimeoutError) as e:
        guarded = safe_rule_sentence_degree(
            text=text,
            degree=degree,
            reason=f"Ollama connection failed, rule based fallback: {e}"
        )
        guarded["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return guarded

    except Exception as e:
        guarded = safe_rule_sentence_degree(
            text=text,
            degree=degree,
            reason=f"LLM response error, rule based fallback: {e}"
        )
        guarded["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return guarded


def build_backend_response(sentence_result: Dict[str, Any], degree_result: Dict[str, Any]) -> Dict[str, Any]:
    semantic_result = semantic_sentence_postprocess(
        sentence_result=sentence_result,
        degree_result=degree_result,
        use_llm=True
    )

    return {
        "mode": "sentence",
        "raw_ai_result": sentence_result,
        "degree_result": degree_result,
        "semantic_llm_result": semantic_result,
        "final_result": {
            "text": semantic_result["final_text"],
            "modified": semantic_result["apply_degree"]
        }
    }


if __name__ == "__main__":
    test_sentence_result = {
        "text": "나는 화났다",
        "confidence": 0.91
    }

    test_degree_result = {
        "degree": "strong",
        "degree_ko": "강함",
        "confidence": 0.82
    }

    result = build_backend_response(test_sentence_result, test_degree_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))