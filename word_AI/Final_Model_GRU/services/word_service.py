import tensorflow as tf
import numpy as np
import os
import json

class GRUWordInference:
    def __init__(self, model_path, label_path, seq_len=30):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 발견 실패: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"라벨 발견 실패: {label_path}")

        print("AI 엔진 및 라벨 맵 로딩 중...")
        self.model = tf.keras.models.load_model(model_path)
        
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.label_map = data.get('idx_to_label', data)
        
        self.seq_len = seq_len
        # 실시간 프레임을 담을 버퍼 (Sliding Window)
        self.frame_buffer = []
        print(f"✅ 준비 완료! (단어 수: {len(self.label_map)}, 입력규격: {seq_len}F x 126D)")

    def _extract_hands_126d(self, full_landmarks):
        """전체 랜드마크에서 손 데이터(75~200)만 추출"""
        # 학습 시 사용된 range(75, 201) 로직 반영
        return full_landmarks[75:201]

    def _normalize_shoulder(self, landmarks_126d, pose_landmarks):
        """학습 시 사용된 어깨 기준 정규화 로직 이식"""
        # pose_landmarks에서 어깨 좌표 추출 (11: 왼쪽어깨, 12: 오른쪽어깨)
        # 각 좌표는 [x, y, c] 구조라고 가정
        l_sh = pose_landmarks[11]
        r_sh = pose_landmarks[12]
        
        # 어깨 중심점 계산
        shoulder_center = (l_sh[:2] + r_sh[:2]) / 2
        # 어깨 너비(Scale) 계산
        shoulder_dist = np.linalg.norm(l_sh[:2] - r_sh[:2]) + 1e-6
        
        # 126D 데이터 변형 (x, y 좌표에서 중심점 빼고 너비로 나누기)
        reshaped = landmarks_126d.reshape(-1, 3)
        reshaped[:, :2] = (reshaped[:, :2] - shoulder_center) / shoulder_dist
        
        return reshaped.flatten()

    def add_frame(self, full_landmarks):
        """실시간으로 들어오는 랜드마크 프레임을 버퍼에 추가"""
        # 1. 전처리 (슬라이싱 및 정규화)
        hands_data = self._extract_hands_126d(full_landmarks)
        # 정규화를 위해 pose 부분(0~74)도 필요하므로 전체 전달
        pose_data = full_landmarks[0:75].reshape(-1, 3)
        processed_frame = self._normalize_shoulder(hands_data, pose_data)
        
        # 2. 버퍼 관리 (Sliding Window)
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)

    def predict(self):
        """버퍼가 꽉 찼을 때 모델 추론 수행"""
        if len(self.frame_buffer) < self.seq_len:
            return None # 데이터 부족

        # 입력을 (1, 30, 126) 형태로 변환
        input_data = np.expand_dims(np.array(self.frame_buffer), axis=0)
        
        # 모델 예측
        prediction = self.model.predict(input_data, verbose=0)
        idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result_text = self.label_map.get(str(idx), "Unknown")
        
        return {
            "text": result_text,
            "confidence": confidence
        }

if __name__ == "__main__":
    # 테스트 코드
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "artifacts", "Final_GRU_HANDS_126D", "models")
    MODEL_PATH = os.path.join(base_path, "best_model.keras")
    LABEL_PATH = os.path.join(base_path, "label_map.json")
    
    try:
        inference = GRUWordInference(MODEL_PATH, LABEL_PATH)
        
        # 가상의 411차원 데이터 30개를 넣어 테스트
        print("가상 데이터로 추론 테스트 중...")
        for _ in range(35):
            dummy_landmarks = np.random.random(411)
            inference.add_frame(dummy_landmarks)
            
        result = inference.predict()
        print(f"✅ 최종 결과: {result}")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")