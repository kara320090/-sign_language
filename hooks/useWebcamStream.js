import { useCallback, useEffect, useRef, useState } from "react";

const CAMERA_STATUS = {
  NOT_REQUESTED: "not_requested",
  REQUESTING: "requesting",
  CONNECTED: "connected",
  PERMISSION_DENIED: "permission_denied",
  DISCONNECTED: "disconnected",
};

const CAMERA_DENIED_MESSAGE =
  "브라우저 주소창 왼쪽의 사이트 설정 또는 카메라 아이콘에서 카메라 권한을 허용으로 변경한 뒤 새로고침해 주세요.";

export function useWebcamStream(enabled) {
  const [stream, setStream] = useState(null);
  const [cameraStatus, setCameraStatus] = useState(CAMERA_STATUS.NOT_REQUESTED);
  const [cameraError, setCameraError] = useState("");
  const [videoReady, setVideoReady] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setVideoReady(false);
    setStream(null);
    setCameraStatus(CAMERA_STATUS.DISCONNECTED);
  }, []);

  const requestCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus(CAMERA_STATUS.PERMISSION_DENIED);
      setCameraError("현재 브라우저에서는 카메라 기능을 사용할 수 없습니다.");
      return;
    }

    setCameraStatus(CAMERA_STATUS.REQUESTING);
    setCameraError("");

    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = nextStream;
      if (videoRef.current) {
        videoRef.current.srcObject = nextStream;
        videoRef.current.muted = true;
        videoRef.current.playsInline = true;
        videoRef.current.play().catch(() => {});
      }
      setStream(nextStream);
      setCameraStatus(CAMERA_STATUS.CONNECTED);
    } catch (error) {
      const denied = error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError";
      setStream(null);
      setCameraStatus(denied ? CAMERA_STATUS.PERMISSION_DENIED : CAMERA_STATUS.DISCONNECTED);
      setCameraError(denied ? CAMERA_DENIED_MESSAGE : "카메라를 연결할 수 없습니다. 다른 앱에서 카메라를 사용 중인지 확인해 주세요.");
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      stopCamera();
      setCameraStatus(CAMERA_STATUS.NOT_REQUESTED);
      return;
    }

    requestCamera();

    return () => {
      stopCamera();
    };
  }, [enabled, requestCamera, stopCamera]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) {
      setVideoReady(false);
      return undefined;
    }

    let mounted = true;

    const markReady = () => {
      if (!mounted) return;
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        setVideoReady(true);
      }
      video.play().catch(() => {});
    };

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    video.addEventListener("loadedmetadata", markReady);
    video.addEventListener("canplay", markReady);
    video.addEventListener("playing", markReady);
    markReady();

    return () => {
      mounted = false;
      video.removeEventListener("loadedmetadata", markReady);
      video.removeEventListener("canplay", markReady);
      video.removeEventListener("playing", markReady);
    };
  }, [stream]);

  return {
    videoRef,
    stream,
    videoReady,
    cameraStatus,
    cameraError,
    requestCamera,
    stopCamera,
    CAMERA_STATUS,
  };
}
