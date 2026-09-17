# SignText Frontend

HandTalker의 React/Vite 기반 수어 분석 화면입니다. 영상 업로드, 웹캠 녹화, 출력 모드 선택과 Backend 결과 표시를 담당합니다.

## 실행

```powershell
cd frontend
npm install
npm run dev
```

Backend 준비와 모델 파일 안내는 [프로젝트 README](../README.md)를 참고하세요. API 주소는 [`src/api/analysisApi.js`](src/api/analysisApi.js)의 `API_BASE_URL`이며 기본값은 `http://127.0.0.1:8000`입니다.

## 현재 API 연결

| 입력 | 처리 |
|---|---|
| 업로드 영상 | `POST /api/predict/upload` |
| 녹화된 웹캠 영상 | 같은 업로드 API에 `webcam-recording.webm` 전송 |
| 정지 이미지 | `/api/predict/webcam-frame` 호출 코드가 있으나 현재 Backend 라우트는 미등록 |
| 영상·이미지 입력 없음 | 일부 웹캠 코드에서 mock 결과 반환 |

키포인트 준비 상태와 화면 시연용 mock 모듈이 남아 있으므로 화면 표시만으로 실제 모델 연결 여부를 판단하지 않고 Backend의 `model_status`를 함께 확인합니다.

## 구조

- `src/api/`: API 호출과 Backend 응답 변환
- `src/components/`: 업로드·웹캠 입력, 결과 패널, 상태 표시
- `src/hooks/`: 분석 상태, 카메라 연결과 자동 분석 흐름
- `src/constants/`: 입력·출력 모드
- `src/mocks/`: 화면·분석 흐름 개발용 대체 데이터
- `src/utils/`: 결과 형식과 상태 문구 변환

배포용 Frontend 빌드는 `npm run build`, 로컬 빌드 미리보기는 `npm run preview`를 사용합니다.
