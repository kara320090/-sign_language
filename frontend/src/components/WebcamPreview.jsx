import { ANALYSIS_STATUS } from "../constants/analysisStatus";

export function WebcamPreview({ videoRef, cameraStatus, cameraError, status }) {
  const denied = status === ANALYSIS_STATUS.CAMERA_DENIED;

  return (
    <div className="preview-frame">
      <video ref={videoRef} className="webcam-video" autoPlay playsInline muted />
      {!denied && cameraStatus !== "connected" && (
        <div className="preview-placeholder">
          <div className="person-mark" />
          <p>카메라를 준비하고 있습니다</p>
        </div>
      )}
      {denied && (
        <div className="permission-panel">
          <strong>카메라 권한이 차단되었습니다.</strong>
          <p>
            {cameraError ||
              "브라우저 주소창 왼쪽의 사이트 설정 또는 카메라 아이콘에서 카메라 권한을 허용으로 변경한 뒤 새로고침해 주세요."}
          </p>
        </div>
      )}
    </div>
  );
}
