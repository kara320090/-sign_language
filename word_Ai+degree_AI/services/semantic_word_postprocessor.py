import json
import requests

class SemanticWordPostprocessor:
    def __init__(self, model_name="qwen2.5:7b"):
        # 로컬 Ollama 서버 주소
        self.url = "http://localhost:11434/api/generate"
        self.model_name = model_name

    def process(self, word_text, degree_ko):
        """
        단어와 감정 강도를 받아 LLM 후처리를 수행합니다.
        예: ("화나다", "강함") -> "정말 화나다"
        """
        # 팀 공지에서 정의한 프롬프트 규격 반영
        prompt = f"""
        수어 단어 인식 결과와 감정의 강도 데이터가 주어집니다. 
        강도를 적절히 반영하여 자연스러운 한국어 단어나 구를 만들어주세요.

        입력 데이터:
        - 인식된 단어: {word_text}
        - 감정 강도: {degree_ko}

        출력은 반드시 아래 JSON 형식만 반환하세요:
        {{
          "apply_degree": true,
          "final_text": "수정된 최종 텍스트",
          "target_expression": "{word_text}",
          "modifier": "추가된 수식어",
          "reason": "반영 이유"
        }}
        """

        try:
            response = requests.post(self.url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            })
            # Ollama로부터 받은 응답(JSON 문자열)을 파싱하여 반환
            return json.loads(response.json().get('response'))
        except Exception as e:
            # Fallback: 서버가 꺼져있거나 에러 시 규칙 기반 반환
            return {
                "apply_degree": False,
                "final_text": f"{word_text} ({degree_ko})",
                "reason": f"LLM 호출 실패: {str(e)}"
            }

if __name__ == "__main__":
    # 테스트 코드
    processor = SemanticWordPostprocessor()
    print("Ollama 후처리 테스트 시작...")
    # 테스트용 데이터 (실제로는 AI 모델들의 결과가 들어옴)
    result = processor.process("슬프다", "강함")
    print(f"✅ 최종 결과: {result}")