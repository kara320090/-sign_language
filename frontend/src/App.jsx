import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppHeader } from "./components/AppHeader";
import { MediaWorkspace } from "./components/MediaWorkspace";
import { ResultPanel } from "./components/ResultPanel";
import { ANALYSIS_STATUS } from "./constants/analysisStatus";
import { INPUT_MODES } from "./constants/inputModes";
import { OUTPUT_MODES } from "./constants/outputModes";
import { useAnalysisFlow } from "./hooks/useAnalysisFlow";
import { useAutoWebcamAnalysis } from "./hooks/useAutoWebcamAnalysis";
import { useKeypointReadiness } from "./hooks/useKeypointReadiness";
import { useWebcamStream } from "./hooks/useWebcamStream";

export default function App() {
  const [inputMode, setInputMode] = useState(INPUT_MODES.WEBCAM);
  const [outputMode, setOutputMode] = useState(OUTPUT_MODES.SENTENCE_DEGREE);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadPlayRequestId, setUploadPlayRequestId] = useState(0);
  const [isUploadVideoEnded, setIsUploadVideoEnded] = useState(false);
  const uploadVideoRef = useRef(null);

  const {
    status,
    setStatus,
    result,
    errorMessage,
    setErrorMessage,
    uploadKeypoints,
    isAnalyzing,
    resetAnalysis,
    runUploadAnalysis,
    runWebcamAnalysis,
  } = useAnalysisFlow();

  const webcam = useWebcamStream(inputMode === INPUT_MODES.WEBCAM);
  const webcamConnected = webcam.cameraStatus === webcam.CAMERA_STATUS.CONNECTED;
  const keypointReadiness = useKeypointReadiness(inputMode === INPUT_MODES.WEBCAM && webcamConnected);

  const previewUrl = useMemo(() => {
    if (!selectedFile) return "";
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (inputMode !== INPUT_MODES.WEBCAM) return;

    if (webcam.cameraStatus === webcam.CAMERA_STATUS.PERMISSION_DENIED) {
      setStatus(ANALYSIS_STATUS.CAMERA_DENIED);
      setErrorMessage(webcam.cameraError);
      return;
    }

    if (webcamConnected && status === ANALYSIS_STATUS.IDLE) {
      setStatus(ANALYSIS_STATUS.CAMERA_READY);
    }
  }, [
    inputMode,
    setErrorMessage,
    setStatus,
    status,
    webcam.CAMERA_STATUS.PERMISSION_DENIED,
    webcam.cameraError,
    webcam.cameraStatus,
    webcamConnected,
  ]);

  const effectiveKeypoints =
    inputMode === INPUT_MODES.WEBCAM ? keypointReadiness.keypoints : uploadKeypoints;

  const handleInputModeChange = useCallback((nextMode) => {
    setInputMode(nextMode);
    resetAnalysis(ANALYSIS_STATUS.IDLE);
    setIsUploadVideoEnded(false);
    if (nextMode === INPUT_MODES.UPLOAD) {
      setSelectedFile(null);
    }
  }, [resetAnalysis]);

  const handleFileChange = useCallback((file) => {
    setSelectedFile(file);
    setIsUploadVideoEnded(false);
    resetAnalysis(ANALYSIS_STATUS.IDLE);
  }, [resetAnalysis]);

  const handleUploadStart = useCallback(() => {
    setIsUploadVideoEnded(false);
    setUploadPlayRequestId((requestId) => requestId + 1);

    if (uploadVideoRef.current) {
      uploadVideoRef.current.currentTime = 0;
      uploadVideoRef.current.play().catch(() => {});
    }

    runUploadAnalysis(selectedFile, outputMode);
  }, [outputMode, runUploadAnalysis, selectedFile]);

  const handleUploadStop = useCallback(() => {
    if (uploadVideoRef.current) {
      uploadVideoRef.current.pause();
      uploadVideoRef.current.currentTime = 0;
    }

    setIsUploadVideoEnded(false);
    resetAnalysis(ANALYSIS_STATUS.IDLE);
  }, [resetAnalysis]);

  const handleUploadEnded = useCallback(() => {
    setIsUploadVideoEnded(true);
  }, []);

  const captureWebcamFrame = useCallback(() => {
    const video = webcam.videoRef.current;

    if (!video || video.readyState < 2) {
      return Promise.resolve(null);
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const context = canvas.getContext("2d");

    if (!context) {
      return Promise.resolve(null);
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        "image/jpeg",
        0.9
      );
    });
  }, [webcam.videoRef]);

  const handleAutoWebcamAnalysis = useCallback(async () => {
    const videoBlob = await recordWebcamVideo(stream, 3000);
    runWebcamAnalysis(outputMode, imageBlob);
  }, [captureWebcamFrame, outputMode, runWebcamAnalysis]);

  useAutoWebcamAnalysis({
    inputMode,
    cameraStatus: webcam.cameraStatus,
    connectedStatus: webcam.CAMERA_STATUS.CONNECTED,
    keypoints: keypointReadiness.keypoints,
    status,
    isAnalyzing,
    setStatus,
    runWebcamAnalysis: handleAutoWebcamAnalysis,
  });

  return (
    <main className="app-shell">
      <AppHeader inputMode={inputMode} onInputModeChange={handleInputModeChange} />

      <div className="workspace-grid">
        <MediaWorkspace
          inputMode={inputMode}
          status={status}
          keypoints={effectiveKeypoints}
          outputMode={outputMode}
          onOutputModeChange={setOutputMode}
          videoRef={webcam.videoRef}
          cameraStatus={webcam.cameraStatus}
          cameraError={webcam.cameraError}
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          uploadVideoRef={uploadVideoRef}
          uploadPlayRequestId={uploadPlayRequestId}
          onFileChange={handleFileChange}
          onUploadStart={handleUploadStart}
          onUploadStop={handleUploadStop}
          onUploadEnded={handleUploadEnded}
          isAnalyzing={isAnalyzing}
        />

        <ResultPanel
          status={status}
          result={result}
          outputMode={outputMode}
          errorMessage={errorMessage}
        />
      </div>
    </main>
  );
}
