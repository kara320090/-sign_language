def apply_semantic_postprocess(
    mode: str,
    text: str,
    degree: str,
    degree_ko: str
) -> dict:
    """
    word/sentence 결과와 degree 결과를 결합하는 규칙 기반 후처리.

    보고서 기준:
    - degree+word: pred_word + pred_degree + modifier
    - degree+sentence: raw_text + degree modifier
    - normal은 원문 유지
    - weak은 '조금'
    - strong은 '매우'
    """

    if not text:
        text = "인식 결과 없음"

    # 모델 결과가 실패/대기 상태면 degree를 붙이지 않음
    invalid_texts = [
        "인식불가",
        "인식 대기",
        "영상 분석 실패",
        "keypoint 추출 성공",
        "문장 인식 실패",
        "통합 모델 오류",
    ]

    if text in invalid_texts:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "인식 결과가 확정되지 않아 표현정도를 반영하지 않았습니다.",
            "processor_status": "rule_fallback",
        }

    # normal은 보고서 기준으로 수식어 없이 원문 유지
    if degree == "normal":
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": "표현정도가 normal이므로 원문을 유지했습니다.",
            "processor_status": "rule_fallback",
        }

    if degree == "weak":
        modifier = "조금"
    elif degree == "strong":
        modifier = "매우"
    else:
        return {
            "apply_degree": False,
            "final_text": text,
            "target_expression": "",
            "modifier": "",
            "reason": f"알 수 없는 degree 값({degree})이므로 원문을 유지했습니다.",
            "processor_status": "rule_fallback",
        }

    # degree를 붙이면 자연스러운 감정/상태 표현들
    emotion_state_terms = [
        "화나다",
        "화났다",
        "아프다",
        "아픕니다",
        "힘들다",
        "힘듭니다",
        "슬프다",
        "슬픕니다",
        "감사합니다",
        "감사",
        "도와주세요",
        "도움이 필요합니다",
        "필요합니다",
        "싫다",
        "싫습니다",
        "좋다",
        "좋습니다",
        "미안합니다",
    ]

    # 문장 안에 감정/상태 표현이 있으면 그 표현 앞에만 수식어 적용
    for term in sorted(emotion_state_terms, key=len, reverse=True):
        if term in text:
            final_text = text.replace(term, f"{modifier} {term}", 1)

            return {
                "apply_degree": True,
                "final_text": final_text,
                "target_expression": term,
                "modifier": modifier,
                "reason": f"감정 또는 상태 표현으로 판단되어 {degree} 표현정도를 반영했습니다.",
                "processor_status": "rule_fallback",
            }

    # 일반 명사/행동 단어는 무리하게 '조금/매우'를 붙이지 않음
    return {
        "apply_degree": False,
        "final_text": text,
        "target_expression": "",
        "modifier": "",
        "reason": "일반 단어 또는 일반 문장으로 판단되어 표현정도를 직접 반영하지 않았습니다.",
        "processor_status": "rule_fallback",
    }