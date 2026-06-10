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
  onUploadEnded,
}) {
  const isUpload = inputMode === INPUT_MODES.UPLOAD;
  const showMediaStatus = status !== ANALYSIS_STATUS.LLM_FALLBACK;

  return (
    <section className="media-workspace" aria-label="수어 입력 영역">
      <div className="media-surface">
        <KeypointOverlay keypoints={keypoints} />
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
            onEnded={onUploadEnded}
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
        <OutputModeSelector value={outputMode} onChange={onOutputModeChange} />
        <div className="start-controls">
          {isUpload && (
            <label className="file-picker">
              파일 선택
              <input accept="video/*" type="file" onChange={(event) => onFileChange(event.target.files?.[0] ?? null)} />
            </label>
          )}
          <StartIndicatorButton
            inputMode={inputMode}
            status={status}
            onUploadStart={onUploadStart}
            onUploadStop={onUploadStop}
            disabled={false}
          />
        </div>
      </div>
    </section>
  );
}
