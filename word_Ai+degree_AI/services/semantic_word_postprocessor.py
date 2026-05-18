import json
import requests
import re

class SemanticWordPostprocessor:
    def __init__(self, model_name="qwen2.5:7b"):
        self.url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        # [피드백 3 반영] 규칙 기반 Fallback 사전
        self.fallback_mapping = {
            "강함": "매우",
            "약함": "조금",
            "보통": ""
        }

    def _clean_json_string(self, raw_str):
        """[피드백 1 반영] LLM 응답에서 마크다운 태그 등을 제거하고 순수 JSON만 추출"""
        # ```json { ... } ``` 형태 제거
        clean_str = re.sub(r'```json|```', '', raw_str).strip()
        return clean_str

    def process(self, word_text, degree_ko):
        """
        단어와 감정 강도를 받아 LLM 후처리를 수행합니다.
        강력한 예외 처리와 검증 로직이 포함되어 있습니다.
        """
        prompt = f"""
        당신은 청각장애인의 수어를 자연스러운 일상 표현으로 보정해주는 '친근한 대화형 수어 통역사'입니다.
        입력된 단어와 감정 강도를 조합하여 [수식어 + 원본 단어] 형태의 구(Phrase) 표현을 만들되, 반드시 아래의 [3단계 프로세스]를 거쳐 검토한 후 최종 결과를 출력하세요.

        [입력 데이터]
        - 인식된 단어: {word_text}
        - 표현의 강도: {degree_ko}

        [⚠️ 필수 준수 규칙]
        1. 문어체(책, 뉴스, 사전에만 나오는 딱딱한 표현)는 절대 금지합니다.
        - ❌ 나쁜 예: 극도로 슬프다, 심각하게 슬프다, 지대하게 기쁘다 (일상 대화에서 쓰지 않음)
        2. 일상 구어체(사람들이 평소에 진짜 말로 주고받는 친근하고 자연스러운 표현)를 사용하세요.
        -  좋은 예: 정말 너무 슬프다, 진짜 많이 슬프다, 너무나도 슬프다
        3. 임의로 맥락을 확장하여 새로운 상황이나 '서술형 문장'을 창작하지 마세요. 오직 강도에 맞는 수식어만 단어 앞에 결합하세요.
        - ❌ 나쁜 예: "혈통" + "강함" -> "혈통 관계 정말 중요해." (문장 창작으로 인한 의미 왜곡)
        -  좋은 예: "혈통" + "강함" -> "진짜 혈통" / "슬프다" + "강함" -> "진짜 너무 슬프다"
        4. JSON 내의 모든 필드값은 100% 한국어로만 작성하세요. 특히 "modifier"에 넣는 수식어는 "final_text" 내부에 실제로 사용한 부사와 글자 하나 틀리지 않고 완벽하게 일치해야 합니다.
        5. 감정 강도가 "보통"인 경우에는 어떠한 수식어(부사)도 결합하는 것을 절대 금지합니다. 억지로 보정을 하려고 조사를 붙이거나 부사를 꾸미지 말고 원래 단어만 그대로 출력하세요.
        - ❌ 나쁜 예: "친구" + "보통" -> "진짜 친구" / "안내원" + "보통" -> "진짜 안내원" (보통 강도인데 억지로 수식어를 붙임)
        -  좋은 예: "친구" + "보통" -> "친구" / "안내원" + "보통" -> "안내원" (수식어 없이 깔끔하게 원본 유지)
        - 강도가 "보통"일 때, "modifier" 필드값은 반드시 단어 대신 "없음"이라는 글자로만 채워야 합니다.

        
        [⚙️ 3단계 처리 프로세스]
        - 1단계 [초안 작성]: 단어와 강도를 조합하여 1차 표현 후보를 마음속으로 작성합니다.
        - 2단계 [비판적 검토]: 자기가 만든 초안을 다시 읽으며 "내가 임의로 뜻을 확장해 문장을 창작하지는 않았는가?", "modifier와 final_text의 부사 글자가 서로 일치하는가?" 비판적으로 자가 검토하세요.
        - 3단계 [수정 및 출력]: 검토를 거쳐 문장 창작과 필드 불일치 오류를 모두 수정한 최종 표현을 JSON 형식으로 반환하세요. 부연 설명은 절대 금지합니다.

        [출력 JSON 규격]
        출력은 반드시 아래 JSON 형식만 반환하세요:
        {{
          "apply_degree": true,
          "final_text": "보정된 최종 표현 ([수식어 + 원본단어] 구조 필수)",
          "target_expression": "{word_text}",
          "modifier": "final_text에 실제 사용된 부사 (final_text와 완벽 일치 필수)",
          "reason": "1단계 초안에서 발생한 문장 창작이나 필드 불일치 모순을 2단계 비판 검토를 통해 어떻게 고쳤는지 구체적으로 기술"
        }}
        """

        try:
            response = requests.post(self.url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }, timeout=15)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            # 1. 원본 응답 추출 및 청소
            raw_response = response.json().get('response', '')
            clean_response = self._clean_json_string(raw_response)
            
            # 2. JSON 파싱
            result = json.loads(clean_response)
            
            # [피드백 2 반영] 필수 Key 검증
            required_keys = ["apply_degree", "final_text", "target_expression", "modifier", "reason"]
            if not all(key in result for key in required_keys):
                raise ValueError("Missing required keys in LLM response")

            # [피드백 4 반영] 성공 상태 추가
            result["status"] = "success"
            return result

        except Exception as e:
            # [피드백 3 반영] 강화된 Fallback 로직
            modifier = self.fallback_mapping.get(degree_ko, "")
            final_text = f"{modifier} {word_text}".strip() if modifier else word_text
            
            return {
                "status": "fallback", # [피드백 4 반영]
                "apply_degree": True if modifier else False,
                "final_text": final_text,
                "target_expression": word_text,
                "modifier": modifier,
                "reason": f"LLM 호출 실패 또는 응답 오류로 인한 규칙 기반 처리 ({str(e)})"
            }

if __name__ == "__main__":
    processor = SemanticWordPostprocessor()
    print("🚀 [고도화 버전] Ollama 후처리 테스트 시작...")
    
    # 테스트 시나리오
    test_word = "슬프다"
    test_degree = "강함"
    
    result = processor.process(test_word, test_degree)
    print(f"\n✅ 최종 결과:\n{json.dumps(result, indent=2, ensure_ascii=False)}")