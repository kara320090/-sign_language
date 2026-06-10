const KEYPOINT_LABELS = [
  { key: "hands", label: "손" },
  { key: "face", label: "얼굴" },
  { key: "pose", label: "신체" },
];

export function KeypointOverlay({ keypoints, liveReady = true }) {
  return (
    <div
      className="keypoint-overlay"
      aria-label={liveReady ? "keypoint 인식 상태" : "keypoint 인식 준비 중"}
      title={liveReady ? "실시간 keypoint 인식 중" : "카메라 영상 준비 중"}
    >
      {KEYPOINT_LABELS.map((item) => (
        <span className={keypoints[item.key] ? "keypoint-chip detected" : "keypoint-chip"} key={item.key}>
          <span className="keypoint-dot" />
          {item.label}
        </span>
      ))}
    </div>
  );
}
