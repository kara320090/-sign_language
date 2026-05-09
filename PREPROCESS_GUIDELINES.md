전처리 가이드 (팀용)
====================

목적
----
이 문서는 프로젝트에서 전처리 기준을 통일하기 위한 팀용 체크리스트입니다. 단어(word) 단위 데이터와 문장(sentence) 단위 데이터가 섞여 있을 때도 동일한 기준으로 처리할 수 있도록 규칙과 예시를 제공합니다.

핵심 원칙
---------
- Keypoint 형식: OpenPose 스타일 JSON (`people` 안에 `pose_keypoints_2d`, `hand_left_keypoints_2d`, `hand_right_keypoints_2d`, 선택적으로 `face_keypoints_2d`)
- 파일명 라벨 규칙: 파일명에 `WORD\d+` 또는 `FS\d+` 형태의 라벨 포함
- 방향 필터: 정면(F)만 사용할 경우 파일명에 `F` 포함 여부로 판단
- 시퀀스 길이: `SEQ_LEN = 30` (균등 리샘플링 또는 패딩)
- 정규화: 어깨 중점(origin) 및 어깨 너비(scale) 기준으로 위치/스케일 정규화
- 결측 처리: confidence<=0 좌표는 (0,0)으로 처리(마스킹)
- 출력: `X.npy`, `y.npy`, `label_map.json`, `used_videos.csv`, `label_counts.csv`, `error_log.csv`, `preprocess_meta.json`

파일명 및 어노테이션 규칙
-------------------------
1) 파일명 포맷 (기본)
   - 패턴: `<video_id>_<frame_num>_keypoints.json`
   - 예: `03OKVS_011_WORD00001_F_0_keypoints.json`
   - 라벨 추출: 정규식 `WORD\d+|FS\d+`로 파일명에서 라벨을 추출

2) 문장(single video contains sentence) 어노테이션(권장)
   - 문장 비디오의 경우 각 단어 구간을 별도 어노테이션 파일로 제공합니다.
   - 권장 CSV 형식 (UTF-8, 헤더 포함):
     `video_uid,start_frame,end_frame,label`
     예:
     `03SENT_001,0,45,WORD00012`
     `03SENT_001,46,90,WORD00005`

   - 또는 JSON 형식:
     {
       "video_uid": "03SENT_001",
       "segments": [
         {"start":0, "end":45, "label":"WORD00012"},
         {"start":46, "end":90, "label":"WORD00005"}
       ]
     }

전처리 흐름 (단계별)
-------------------
1) 스캔
   - ZIP 모드: ZIP 내부 `*_keypoints.json` 멤버를 순회하여 파일명에서 video_id/frame/label을 파싱
   - 폴더 모드: 추출된 `extracted_keypoints_front` 디렉토리를 rglob로 스캔

2) 프레임 레벨 피처 추출
   - `pose`, `hand_left`, `hand_right` 키포인트를 추출
   - `reshape_keypoints`, `fix_length`로 고정 길이(flat)로 변환
   - `get_origin_and_scale`로 origin 및 scale 계산
   - `normalize_points`로 좌표 정규화

3) 시퀀스 구성(샘플 만들기)
   - 단어 단위: 파일명(또는 폴더 구조)에서 단어별 프레임들을 모아 하나의 샘플로 구성
   - 문장 단위: 어노테이션 CSV/JSON의 segments를 읽어 각 segment를 하나의 샘플(단어)로 변환
   - segment가 프레임 단위가 아닌 경우(초 단위), 프레임으로 변환 필요

4) 리샘플링/패딩
   - 길이가 `SEQ_LEN`보다 길면 균등 인덱 샘플링(예: np.linspace 0..L-1, seq_len개)
   - 짧으면 뒤쪽에 제로 패딩

5) 라벨 맵 생성
   - `label_to_idx`, `idx_to_label`을 생성하여 `label_map.json`로 저장

6) 결과 저장
   - `X.npy` (float32, shape=(N, SEQ_LEN, feature_dim))
   - `y.npy` (int64, shape=(N,)) 또는 문장-시퀀스의 경우 별도 시퀀스 라벨 파일
   - `used_videos.csv`, `label_counts.csv`, `error_log.csv`, `preprocess_meta.json`

문장(Sequence) 데이터 처리 권장 방법
---------------------------------
옵션 A (권장, 간단): 문장 어노테이션으로부터 각 단어 segment를 잘라서 기존 단어-단위 파이프라인(`01 전처리.py`)에 투입
  - 장점: 기존 코드 변경 최소, 동일한 출력 포맷(X.npy, y.npy)
  - y: 각 샘플은 단일 라벨

옵션 B (고급): 문장 전체를 하나의 샘플로 하고 라벨을 라벨 시퀀스로 저장(시퀀스-투-시퀀스 학습 필요)
  - 출력: `X_seq.npy` (N, SEQ_LEN, feature_dim), `y_seq.json` 또는 `y_seq.npy`(패딩된 정수 시퀀스, -1 또는 mask 인덱스 사용)
  - 모델: CTC 또는 seq2seq 디코더 필요
  - 권장 시나리오: 문장 경계가 자연스럽고 단어 속도가 다양할 때

검증 체크리스트 (팀원 전처리 후 확인)
---------------------------------
1. `preprocess_meta.json` 존재 및 핵심 필드 일치 (`seq_len`, `use_face`, `front_only`, `zip_files`)
2. `X.npy`/`y.npy`의 샘플 수가 `used_videos.csv`와 일치
3. `label_map.json`에 사용된 모든 라벨 포함
4. `error_log.csv` 확인: 심각한 오류가 0인지 검토
5. 랜덤 시드(`RANDOM_STATE`)로 데이터 분할 재현 확인

실행 예시
---------
전처리만 실행:
```bash
python "00 실행 런처.py" --stage preprocess
```

추출 스크립트(앞면만 추출):
```bash
python extract_front_only.py
```

권장 워크플로
-------------
1. 모든 팀원이 동일한 `PREPROCESS_GUIDELINES.md`와 `00 config.py`의 핵심 설정(`SEQ_LEN`, `USE_FACE`, `FRONT_ONLY`)을 숙지
2. 문장 데이터가 있으면 **annotation CSV/JSON**을 반드시 팀 스펙에 맞춰 공유
3. 각자 전처리 실행 후 `preprocess_meta.json`을 PR에 첨부하거나 결과 파일의 해시를 공유
4. 기준 성능(예: baseline 모델 정확도)을 기록하고, 얼굴 포함 여부나 다른 변경 사항은 A/B 테스트로 검증

문의 및 변경 제안
-----------------
이 문서는 팀 합의 후 `README`로 병합하세요. 필요하면 제가 `preprocess_config.json` 템플릿과 작은 검증 스크립트를 추가해 드리겠습니다.

호환성 및 비구속성 (Sentence 팀 안내)
-----------------------------------
이 문서는 팀의 공통 기준을 제안하는 것이고, 문장 데이터를 처리하는 팀이 다른 파이프라인을 사용하더라도 괜찮습니다. 다만 팀 간 협업과 모델 공동 실험을 위해 최소한 다음 인터페이스는 맞춰주기를 권장합니다. 이는 문장팀이 기존 `01 전처리.py` 코드를 사용하지 않아도 호환성을 보장하기 위한 약속입니다.

1) 입력/어노테이션
   - 권장: 세그먼트 CSV 또는 JSON (각 행/객체에 `video_uid`, `start_frame`, `end_frame`, `label` 포함).
   - 프레임 단위 대신 초 단위를 쓸 경우(예: 초·밀리초), 꼭 프레임으로 변환된 매핑을 함께 제공.

2) 라벨 맵 (`label_map.json`) 최소 스펙
   - 필수 필드:
     - `label_to_idx`: {"WORD00001": 0, ...}
     - `idx_to_label`: {"0": "WORD00001", ...}
   - 이 맵이 있으면 서로 다른 파이프라인에서 생성된 `y` 인덱스가 일관성을 가집니다.

3) 전처리 메타 (`preprocess_meta.json`) 최소 키
   - 포함 권장 키: `seq_len`, `use_face`, `front_only`, `zip_files` (혹은 `source_mode`/`annotation_file`)
   - 이 파일을 통해 어떤 설정으로 샘플이 만들어졌는지 바로 이해할 수 있어 실험 재현이 쉬워집니다.

4) X/y 출력 포맷(선택적 호환)
   - 만약 문장팀에서 중앙 학습 파이프라인으로 데이터(샘플)를 직접 제공하려면 다음을 맞춰주세요:
     - `X.npy`: float32, shape=(N, SEQ_LEN, feature_dim), 정규화(어깨 기준) 적용
     - `y.npy`: int64, shape=(N,), 라벨은 `label_to_idx` 기준
   - 대안: 문장팀은 세그먼트 CSV만 제공하고, 메인 파이프라인에서 세그먼트를 읽어 샘플을 구성하도록 할 수 있습니다(옵션 A 방식).

5) 오류·메타 공유
   - `error_log.csv` (읽기 실패/파싱 실패 목록), `used_videos.csv`(선택된 샘플 목록) 등은 팀 간 공유 권장.

6) 정규화 규격 문서화
   - 만약 문장팀이 자체 전처리를 수행한다면, 좌표 정규화 방식(origin/scale 계산 방식)과 confidence 처리 규칙을 문서화 해주세요. (예: 어깨 중점 사용, shoulder_width로 나눔, confidence<=0이면 0으로 마킹)

7) 재현성
   - 데이터 분할용 `RANDOM_STATE` 값과 샘플링 세부 규칙(리샘플링 인덱 계산 방법)을 공유하면 실험 비교가 쉬워집니다.

권장 워크플로 (문장팀과의 협업)
--------------------------------
- 문장팀이 세그먼트 CSV를 생성하면, 메인 파이프라인에서 해당 CSV로부터 단어 단위 샘플을 자동으로 만들 수 있도록 커넥터(작은 스크립트)를 작성합니다.
- 또는 문장팀이 직접 `X.npy`/`y.npy`를 제공하는 경우, `label_map.json`과 `preprocess_meta.json`을 반드시 함께 제출합니다.
- 변경 사항(예: 다른 정규화 방법)을 적용한 경우 A/B 테스트 계획과 함께 결과를 비교해 주세요.

요약: 문장팀의 자유를 보장하되, 위 최소 인터페이스를 지켜주면 팀 전체 실험을 일관되게 비교할 수 있습니다.
