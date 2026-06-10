# SignText Frontend

React + Vite 기반 수어 텍스트 변환 웹 UI입니다.

## Folder Structure

```text
frontend/
  package.json
  vite.config.js
  index.html
  src/
    main.jsx
    App.jsx
    styles.css

    api/
      analysisApi.js

    constants/
      inputModes.js
      outputModes.js
      analysisStatus.js

    components/
      AppHeader.jsx
      InputModeTabs.jsx
      MediaWorkspace.jsx
      WebcamPreview.jsx
      UploadPreview.jsx
      KeypointOverlay.jsx
      OutputModeSelector.jsx
      ResultPanel.jsx
      StatusBadge.jsx
      StartIndicatorButton.jsx

    hooks/
      useAnalysisFlow.js
      useAutoWebcamAnalysis.js
      useWebcamStream.js
      useKeypointReadiness.js

    mocks/
      mockAnalysisResult.js
      mockKeypointDetector.js

    utils/
      formatConfidence.js
      getStatusLabel.js
      getDegreeLabel.js
```

## Current Behavior

- 웹캠 모드에서는 카메라 권한을 요청하고, 손/얼굴/신체 keypoint가 모두 인식되면 자동으로 분석을 시작합니다.
- 웹캠 모드의 `시작` 버튼은 클릭 트리거가 아니라 현재 자동 분석 상태를 표시합니다.
- 업로드 모드에서는 파일을 선택한 뒤 `시작` 버튼으로 분석을 실행합니다.
- 현재 AI 분석과 keypoint 감지는 mock으로 동작하며, 실제 API/MediaPipe 연결 시 `api/`와 `hooks/` 내부만 교체하면 됩니다.

## Commands

```bash
npm install
npm run dev
npm run build
```
