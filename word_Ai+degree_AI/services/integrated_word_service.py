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
    def __init__(self, model_path, label_path, mapping_path, degree_model_path):
        import joblib  # scikit-learn 기반 감정 표현 모델(.joblib) 로드용 라이브러리 추가
        
        print("🚀 [통합 서비스] 엔진 초기화 중...")
        
        # [안전장치] 팀원의 감정 표현 모델(.joblib) 검사 항목을 file_checks에 추가합니다.
        file_checks = [
            ("단어 모델(.keras)", model_path), 
            ("라벨 매핑(.json)", label_path), 
            ("한국어 사전(.csv)", mapping_path),
            ("감정 표현 모델(.joblib)", degree_model_path)  # 신규 추가
        ]
        
        for name, path in file_checks:
            if not os.path.exists(path):
                print(f"❌ [경로 에러] 필수 {name} 파일이 지정된 위치에 없습니다!")
                print(f"🔎 컴퓨터가 찾아간 주소 -> {path}")
                raise FileNotFoundError(f"{name} 파일 분실")

        # 검사가 통과되면 실제 엔진 가동
        # 1. 단어 인식 AI 엔진 가동 (TensorFlow)
        self.word_engine = GRUWordInference(model_path, label_path, mapping_path)
        
        # 2. [수정 완료] 팀원의 bundle 박스를 열고 진짜 모델(MLPClassifier) 알맹이만 적출
        loaded_bundle = joblib.load(degree_model_path)
        self.degree_model = loaded_bundle["model"]  # "model" Key를 사용해 진짜 인공지능 객체 바인딩
        
        # 3. 의미 보정 LLM 후처리 엔진 가동 (Ollama)
        self.llm_engine = SemanticWordPostprocessor()
        
        print("✨ [단어 GRU + 감정 MLP + Ollama LLM] 모든 실제 AI 부품 결합 및 로드 완료.")

    # 수정 핵심: 이제 매개변수에서 외부 의존성 글자인 degree_ko를 삭제합니다.
    def process_realtime(self, full_landmarks):
        """
        [1차 좌표 수신] ➔ [단어 GRU 시퀀스 누적 및 추론] + [감정 MLP 1F 단일 추론] ➔ [LLM 최종 문장 보정]
        """
        # 1. 단어 AI 버퍼(시계열 30F)에 현재 프레임 누적 및 예측
        self.word_engine.add_frame(full_landmarks)
        word_result = self.word_engine.predict()
        
        # 2. 단어 AI 임계값 필터 통과로 수어 단어가 완벽히 잡히는 '결정적 타이밍'에 진입했다면?
        if word_result and word_result['status'] == 'success':
            
            # 3. 감정 AI 규격 연동: 전체 좌표(411D) 중 감정 전용 앞단 280차원 슬라이싱 및 형상 변환 (1, 280)
            degree_input = full_landmarks[:280].reshape(1, -1)
            
            # 4. 실제 결합된 팀원의 감정 표현 AI 모델 추론 가동 (결과 예: 0, 1, 2)
            degree_pred = self.degree_model.predict(degree_input)[0]
            
            # 5. 모델 숫자 결과를 시스템 표준 문자열("약함", "보통", "강함")로 매핑
            degree_map = {0: "약함", 1: "보통", 2: "강함"}
            real_degree_ko = degree_map.get(int(degree_pred), "보통")
            
            # 6. 두 인공지능이 뽑아낸 교차 결과(진짜 단어텍스트 + 진짜 감정강도)를 LLM 후처리 엔진에 전달
            final_output = self.llm_engine.process(word_result['text'], real_degree_ko)
            
            return {
                "status": "success",
                "original_word": word_result['text'],
                "confidence": word_result['confidence'],
                "predicted_degree": real_degree_ko,  # 100% 인공지능이 실시간 계산한 진짜 감정 수치
                "final_sentence": final_output['final_text'],
                "modifier": final_output['modifier'],
                "reason": final_output['reason']
            }
        
        return {"status": "processing", "message": "데이터 축적 중이거나 확신도가 낮음"}

if __name__ == "__main__":
    base_model_path = PROJECT_ROOT / "word_AI" / "Final_Model_GRU" / "artifacts" / "Final_GRU_HANDS_126D" / "models"
    
    MODEL = str(base_model_path / "best_model.keras")
    LABEL = str(base_model_path / "label_map.json")
    MAP = str(base_model_path / "word_label_mapping.csv")
    
    DEGREE_MODEL = str(PROJECT_ROOT / "degree_AI" / "models" / "degree_frame_280d_anger_mlp.joblib")

    try:
        # 서비스 생성 및 구동 (수정된 인프라 스펙에 맞게 감정 모델 경로까지 4개 인자 모두 주입)
        service = IntegratedWordService(MODEL, LABEL, MAP, DEGREE_MODEL)
        
        print("\n🔥 [실제 부품 전면 결합] 최종 실시간 통합 추론 테스트 시작...")
        service.word_engine.threshold = 0.0 # 테스트용 강제 통과
        
        # 35프레임 가상 데이터 주입 (단어 GRU는 축적하고, 감정 MLP는 이 좌표로부터 즉시 강도 연산 수행)
        for _ in range(35):
            dummy_data = np.random.random(411)
            service.process_realtime(dummy_data)
            
        # 마지막 프레임 주입 (★중요: 이제 외부에서 degree_ko="강함"을 강제로 넣지 않고 가상 데이터만 던집니다)
        final_res = service.process_realtime(np.random.random(411))
        
        import json
        print(f"\n✅ 통합 서비스 최종 응답:\n{json.dumps(final_res, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"\n❌ 초기화 및 실행 중 최종 예외 발생: {e}")