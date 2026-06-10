import { ANALYSIS_STATUS } from "../constants/analysisStatus";

export function getStatusLabel(status) {
  const labels = {
    [ANALYSIS_STATUS.IDLE]: "대기 중",
    [ANALYSIS_STATUS.CAMERA_READY]: "카메라 준비",
    [ANALYSIS_STATUS.WAITING_KEYPOINTS]: "인식 대기 중",
    [ANALYSIS_STATUS.LOADING]: "분석 중",
    [ANALYSIS_STATUS.SUCCESS]: "분석 완료",
    [ANALYSIS_STATUS.ERROR]: "분석 실패",
    [ANALYSIS_STATUS.CAMERA_DENIED]: "카메라 권한 거부",
    [ANALYSIS_STATUS.LLM_FALLBACK]: "LLM fallback",
  };

  return labels[status] ?? "대기 중";
}
