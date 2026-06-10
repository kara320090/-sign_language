import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { INPUT_MODES } from "../constants/inputModes";

function getWebcamLabel(status) {
  if (status === ANALYSIS_STATUS.LOADING) return "자동 분석 중";
  if (status === ANALYSIS_STATUS.SUCCESS || status === ANALYSIS_STATUS.LLM_FALLBACK) return "자동 분석 완료";
  if (status === ANALYSIS_STATUS.CAMERA_DENIED) return "권한 필요";
  if (status === ANALYSIS_STATUS.ERROR) return "자동 분석 오류";
  return "인식 대기 중";
}

function getUploadLabel(status) {
  if (status === ANALYSIS_STATUS.LOADING) return "중단";
  return "시작";
}

export function StartIndicatorButton({ inputMode, status, onUploadStart, onUploadStop, disabled }) {
  const isWebcam = inputMode === INPUT_MODES.WEBCAM;
  const isUploadRunning = !isWebcam && status === ANALYSIS_STATUS.LOADING;
  const label = isWebcam ? getWebcamLabel(status) : getUploadLabel(status);
  const active = isWebcam
    ? status === ANALYSIS_STATUS.LOADING ||
      status === ANALYSIS_STATUS.SUCCESS ||
      status === ANALYSIS_STATUS.LLM_FALLBACK
    : isUploadRunning;

  return (
    <button
      className={active ? "start-button active" : "start-button"}
      type="button"
      disabled={isWebcam || disabled}
      onClick={isUploadRunning ? onUploadStop : onUploadStart}
      aria-live="polite"
    >
      {label}
    </button>
  );
}
