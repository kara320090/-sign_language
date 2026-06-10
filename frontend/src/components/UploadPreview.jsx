import { useEffect } from "react";

export function UploadPreview({ previewUrl, file, videoRef, playRequestId, onEnded }) {
  useEffect(() => {
    if (!playRequestId || !videoRef.current) return undefined;

    const video = videoRef.current;

    const playFromStart = () => {
      video.muted = true;
      video.currentTime = 0;
      video.play().catch(() => {});
    };

    if (video.readyState >= 2) {
      playFromStart();
      return undefined;
    }

    video.addEventListener("loadeddata", playFromStart, { once: true });
    return () => {
      video.removeEventListener("loadeddata", playFromStart);
    };
  }, [playRequestId, videoRef]);

  return (
    <div className="preview-frame">
      {previewUrl ? (
        <video
          className="upload-video"
          src={previewUrl}
          controls
          muted
          preload="auto"
          ref={videoRef}
          onEnded={onEnded}
        />
      ) : (
        <div className="preview-placeholder">
          <div className="person-mark" />
          <p>분석할 수어 영상을 선택해 주세요</p>
        </div>
      )}
      {file && <span className="file-name">{file.name}</span>}
    </div>
  );
}
