import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { INPUT_MODES } from "../constants/inputModes";

function getWebcamLabel(status) {
  if (status === ANALYSIS_STATUS.RECORDING) return "5초 녹화 중";
  if (status === ANALYSIS_STATUS.LOADING) return "모델 분석 중";
  if (status === ANALYSIS_STATUS.SUCCESS || status === ANALYSIS_STATUS.LLM_FALLBACK) return "다시 분석";
  if (status === ANALYSIS_STATUS.CAMERA_DENIED) return "권한 필요";
  if (status === ANALYSIS_STATUS.ERROR) return "다시 시도";
  return "시작";
}

function getUploadLabel(status) {
  if (status === ANALYSIS_STATUS.LOADING) return "중단";
  return "시작";
}

export function StartIndicatorButton({
  inputMode,
  status,
  onUploadStart,
  onUploadStop,
  onWebcamStart,
  disabled,
}) {
  const isWebcam = inputMode === INPUT_MODES.WEBCAM;
  const isUploadRunning = !isWebcam && status === ANALYSIS_STATUS.LOADING;
  const label = isWebcam ? getWebcamLabel(status) : getUploadLabel(status);
  const active = isWebcam
    ? status === ANALYSIS_STATUS.RECORDING ||
      status === ANALYSIS_STATUS.LOADING ||
      status === ANALYSIS_STATUS.SUCCESS ||
      status === ANALYSIS_STATUS.LLM_FALLBACK
    : isUploadRunning;

  return (
    <button
      className={active ? "start-button active" : "start-button"}
      type="button"
      disabled={disabled}
      onClick={
        isWebcam
          ? onWebcamStart
          : isUploadRunning
            ? onUploadStop
            : onUploadStart
      }
      aria-live="polite"
    >
      {label}
    </button>
  );
}
