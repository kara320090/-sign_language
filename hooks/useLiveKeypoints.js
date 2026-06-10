import { useEffect, useState } from "react";
import { detectKeypointSnapshot } from "../api/keypointApi";


const EMPTY_KEYPOINTS = {
  hands: false,
  face: false,
  pose: false,
};

const SNAPSHOT_INTERVAL_MS = 300;
const SNAPSHOT_MAX_WIDTH = 480;


function captureVideoSnapshot(video, canvas) {
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return Promise.resolve(null);
  }

  const scale = Math.min(1, SNAPSHOT_MAX_WIDTH / video.videoWidth);
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

  const context = canvas.getContext("2d");
  if (!context) return Promise.resolve(null);

  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.72);
  });
}


export function useLiveKeypoints(videoRef, enabled, resetKey) {
  const [liveKeypoints, setLiveKeypoints] = useState(EMPTY_KEYPOINTS);

  useEffect(() => {
    setLiveKeypoints(EMPTY_KEYPOINTS);

    if (!enabled) return undefined;

    let stopped = false;
    let timerId = null;
    let controller = null;
    const canvas = document.createElement("canvas");

    const scheduleNext = () => {
      timerId = window.setTimeout(detectCurrentFrame, SNAPSHOT_INTERVAL_MS);
    };

    const detectCurrentFrame = async () => {
      const video = videoRef?.current;

      try {
        const imageBlob = await captureVideoSnapshot(video, canvas);
        if (!imageBlob) {
          if (!stopped) scheduleNext();
          return;
        }

        controller?.abort();
        controller = new AbortController();

        const nextKeypoints = await detectKeypointSnapshot({
          imageBlob,
          signal: controller.signal,
        });

        if (!stopped) {
          setLiveKeypoints(nextKeypoints);
        }
      } catch (error) {
        if (!stopped && error?.name !== "AbortError") {
          console.warn("Live keypoint detection failed.", error);
        }
      } finally {
        if (!stopped) scheduleNext();
      }
    };

    detectCurrentFrame();

    return () => {
      stopped = true;
      if (timerId) window.clearTimeout(timerId);
      controller?.abort();
    };
  }, [enabled, resetKey, videoRef]);

  return liveKeypoints;
}
