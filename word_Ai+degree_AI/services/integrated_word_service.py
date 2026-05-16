import sys
import os
import numpy as np
from pathlib import Path

# 1. 현재 파일 위치를 기준으로 프로젝트 대문 주소('99. 깃헙 코드 -sign_language')를 잡습니다.
CURRENT_DIR = Path(__file__).resolve().parent  # .../word_Ai+degree_AI/services
PROJECT_ROOT = CURRENT_DIR.parent.parent       # .../99. 깃헙 코드 -sign_language

# 2. 준혁님이 명시해주신 단어 AI 서비스 폴더 주소를 정확하게 조립합니다.
WORD_SERVICES_DIR = PROJECT_ROOT / "word_AI" / "Final_Model_GRU" / "services"

# 파이썬 주소록에 등록 (services 폴더명 충돌 방지용)
if str(WORD_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(WORD_SERVICES_DIR))

# 부품 소스코드 임포트
try:
    from word_service import GRUWordInference
    from semantic_word_postprocessor import SemanticWordPostprocessor
except ImportError as e:
    print(f"❌ 소스코드 임포트 실패! 폴더명이 정확한지 확인하세요.")
    print(f"🔎 시스템이 찾으려고 시도한 주소: {WORD_SERVICES_DIR}")
    sys.exit(1)

class IntegratedWordService:
    def __init__(self, model_path, label_path, mapping_path):
        print("🚀 [통합 서비스] 엔진 초기화 중...")
        
        # [안전장치] 파일이 실제로 하드디스크에 있는지 먼저 검사해서 알려주기
        file_checks = [("모델(.keras)", model_path), ("라벨(.json)", label_path), ("매핑(.csv)", mapping_path)]
        for name, path in file_checks:
            if not os.path.exists(path):
                print(f"❌ [경로 에러] 필수 {name} 파일이 지정된 위치에 없습니다!")
                print(f"🔎 컴퓨터가 찾아간 주소 -> {path}")
                raise FileNotFoundError(f"{name} 파일 분실")

        # 검사가 통과되면 실제 엔진 가동
        self.word_engine = GRUWordInference(model_path, label_path, mapping_path)
        self.llm_engine = SemanticWordPostprocessor()
        print("✨ 모든 엔진이 성공적으로 로드되었습니다.")

    def process_realtime(self, full_landmarks, degree_ko="보통"):
        """
        좌표 데이터를 입력받아 [전처리 -> 단어인식 -> LLM 보정] 과정을 수행
        """
        self.word_engine.add_frame(full_landmarks)
        word_result = self.word_engine.predict()
        
        if word_result and word_result['status'] == 'success':
            final_output = self.llm_engine.process(word_result['text'], degree_ko)
            
            return {
                "status": "success",
                "original_word": word_result['text'],
                "confidence": word_result['confidence'],
                "final_sentence": final_output['final_text'],
                "modifier": final_output['modifier'],
                "reason": final_output['reason']
            }
        
        return {"status": "processing", "message": "데이터 축적 중이거나 확신도가 낮음"}

if __name__ == "__main__":
    # 3. 준혁님의 실제 물리 주소에 맞춘 artifacts 경로 조립
    base_model_path = PROJECT_ROOT / "word_AI" / "Final_Model_GRU" / "artifacts" / "Final_GRU_HANDS_126D" / "models"
    
    MODEL = str(base_model_path / "best_model.keras")
    LABEL = str(base_model_path / "label_map.json")
    MAP = str(base_model_path / "word_label_mapping.csv")

    try:
        # 서비스 생성 및 구동
        service = IntegratedWordService(MODEL, LABEL, MAP)
        
        print("\n🔥 [최종 검증] 통합 추론 테스트 시작...")
        service.word_engine.threshold = 0.0 # 테스트용 강제 통과
        
        # 35프레임 가상 데이터 주입
        for _ in range(35):
            dummy_data = np.random.random(411)
            service.process_realtime(dummy_data)
            
        # 마지막 프레임과 함께 "강함" 감정 전달
        final_res = service.process_realtime(np.random.random(411), degree_ko="강함")
        
        import json
        print(f"\n✅ 통합 서비스 최종 응답:\n{json.dumps(final_res, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"\n❌ 초기화 및 실행 중 최종 예외 발생: {e}")