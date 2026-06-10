import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { getStatusLabel } from "../utils/getStatusLabel";

const TONE_BY_STATUS = {
  [ANALYSIS_STATUS.IDLE]: "neutral",
  [ANALYSIS_STATUS.CAMERA_READY]: "neutral",
  [ANALYSIS_STATUS.RECORDING]: "recording",
  [ANALYSIS_STATUS.LOADING]: "loading",
  [ANALYSIS_STATUS.SUCCESS]: "success",
  [ANALYSIS_STATUS.ERROR]: "error",
  [ANALYSIS_STATUS.CAMERA_DENIED]: "error",
  [ANALYSIS_STATUS.LLM_FALLBACK]: "fallback",
};

export function StatusBadge({ status }) {
  const tone = TONE_BY_STATUS[status] ?? "neutral";

  return (
    <span className={`status-badge ${tone}`}>
      <span className="status-dot" />
      {getStatusLabel(status)}
    </span>
  );
}
