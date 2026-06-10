const KEYPOINT_LABELS = [
  { key: "hands", label: "손" },
  { key: "face", label: "얼굴" },
  { key: "pose", label: "신체" },
];

export function KeypointOverlay({ keypoints }) {
  return (
    <div className="keypoint-overlay" aria-label="keypoint 인식 상태">
      {KEYPOINT_LABELS.map((item) => (
        <span className={keypoints[item.key] ? "keypoint-chip detected" : "keypoint-chip"} key={item.key}>
          <span className="keypoint-dot" />
          {item.label}
        </span>
      ))}
    </div>
  );
}
