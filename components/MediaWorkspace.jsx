import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { INPUT_MODES } from "../constants/inputModes";
import { KeypointOverlay } from "./KeypointOverlay";
import { OutputModeSelector } from "./OutputModeSelector";
import { StartIndicatorButton } from "./StartIndicatorButton";
import { StatusBadge } from "./StatusBadge";
import { UploadPreview } from "./UploadPreview";
import { WebcamPreview } from "./WebcamPreview";

export function MediaWorkspace({
  inputMode,
  status,
  keypoints,
  keypointsLiveReady,
  outputMode,
  onOutputModeChange,
  videoRef,
  cameraStatus,
  cameraError,
  selectedFile,
  previewUrl,
  uploadVideoRef,
  uploadPlayRequestId,
  onFileChange,
  onUploadStart,
  onUploadStop,
  onWebcamStart,
  isAnalyzing,
  controlsDisabled,
}) {
  const isUpload = inputMode === INPUT_MODES.UPLOAD;
  const showMediaStatus = status !== ANALYSIS_STATUS.LLM_FALLBACK;

  return (
    <section className="media-workspace" aria-label="수어 입력 영역">
      <div className="media-surface">
        <KeypointOverlay keypoints={keypoints} liveReady={keypointsLiveReady} />
        {showMediaStatus && (
          <div className="media-status">
            <StatusBadge status={status} />
          </div>
        )}
        {isUpload ? (
          <UploadPreview
            previewUrl={previewUrl}
            file={selectedFile}
            videoRef={uploadVideoRef}
            playRequestId={uploadPlayRequestId}
          />
        ) : (
          <WebcamPreview
            videoRef={videoRef}
            cameraStatus={cameraStatus}
            cameraError={cameraError}
            status={status}
          />
        )}
      </div>

      <div className="workspace-controls">
        <OutputModeSelector
          value={outputMode}
          onChange={onOutputModeChange}
          disabled={controlsDisabled}
        />
        <div className="start-controls">
          {isUpload && (
            <label className="file-picker">
              파일 선택
              <input
                accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
                type="file"
                disabled={controlsDisabled}
                onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
              />
            </label>
          )}
          <StartIndicatorButton
            inputMode={inputMode}
            status={status}
            onUploadStart={onUploadStart}
            onUploadStop={onUploadStop}
            onWebcamStart={onWebcamStart}
            disabled={
              isUpload
                ? !selectedFile && status !== ANALYSIS_STATUS.LOADING
                : isAnalyzing ||
                  status === ANALYSIS_STATUS.RECORDING ||
                  cameraStatus !== "connected"
            }
          />
        </div>
      </div>
    </section>
  );
}
