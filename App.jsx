import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppHeader } from "./components/AppHeader";
import { MediaWorkspace } from "./components/MediaWorkspace";
import { ResultPanel } from "./components/ResultPanel";
import { ANALYSIS_STATUS } from "./constants/analysisStatus";
import { INPUT_MODES } from "./constants/inputModes";
import { OUTPUT_MODES } from "./constants/outputModes";
import { useAnalysisFlow } from "./hooks/useAnalysisFlow";
import { useLiveKeypoints } from "./hooks/useLiveKeypoints";
import { useWebcamStream } from "./hooks/useWebcamStream";
import { recordWebcamClip } from "./utils/recordWebcamClip";

export default function App() {
  const [inputMode, setInputMode] = useState(INPUT_MODES.WEBCAM);
  const [outputMode, setOutputMode] = useState(OUTPUT_MODES.SENTENCE_DEGREE);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadPlayRequestId, setUploadPlayRequestId] = useState(0);
  const uploadVideoRef = useRef(null);

  const {
    status,
    setStatus,
    result,
    errorMessage,
    setErrorMessage,
    analysisKeypoints,
    isAnalyzing,
    resetAnalysis,
    runUploadAnalysis,
    runWebcamAnalysis,
  } = useAnalysisFlow();

  const webcam = useWebcamStream(inputMode === INPUT_MODES.WEBCAM);
  const webcamConnected = webcam.cameraStatus === webcam.CAMERA_STATUS.CONNECTED;
  const interactionLocked =
    isAnalyzing || status === ANALYSIS_STATUS.RECORDING;
  const liveKeypointVideoRef =
    inputMode === INPUT_MODES.WEBCAM ? webcam.videoRef : uploadVideoRef;
  const liveKeypointsEnabled =
    inputMode === INPUT_MODES.WEBCAM
      ? webcamConnected && webcam.videoReady
      : Boolean(selectedFile);
  const liveKeypointResetKey = [
    inputMode,
    webcam.videoReady ? "video-ready" : "video-waiting",
    selectedFile?.name ?? "",
    selectedFile?.lastModified ?? "",
  ].join(":");
  const liveKeypoints = useLiveKeypoints(
    liveKeypointVideoRef,
    liveKeypointsEnabled,
    liveKeypointResetKey,
  );
  const displayedKeypoints = liveKeypointsEnabled ? liveKeypoints : analysisKeypoints;
  const keypointLiveReady =
    inputMode === INPUT_MODES.WEBCAM ? webcam.videoReady : Boolean(selectedFile);

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

  const handleInputModeChange = useCallback((nextMode) => {
    if (interactionLocked) return;

    setInputMode(nextMode);
    resetAnalysis(ANALYSIS_STATUS.IDLE);
    if (nextMode === INPUT_MODES.UPLOAD) {
      setSelectedFile(null);
    }
  }, [interactionLocked, resetAnalysis]);

  const handleFileChange = useCallback((file) => {
    if (interactionLocked) return;

    setSelectedFile(file);
    resetAnalysis(ANALYSIS_STATUS.IDLE);
  }, [interactionLocked, resetAnalysis]);

  const handleUploadStart = useCallback(() => {
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

    resetAnalysis(ANALYSIS_STATUS.IDLE);
  }, [resetAnalysis]);

  const handleWebcamStart = useCallback(async () => {
    if (!webcam.stream) {
      setStatus(ANALYSIS_STATUS.ERROR);
      setErrorMessage("웹캠이 연결된 뒤 다시 시도해 주세요.");
      return;
    }

    setStatus(ANALYSIS_STATUS.RECORDING);
    setErrorMessage("");

    try {
      const videoBlob = await recordWebcamClip(webcam.stream);
      await runWebcamAnalysis(videoBlob, outputMode);
    } catch (error) {
      setStatus(ANALYSIS_STATUS.ERROR);
      setErrorMessage(error?.message ?? "웹캠 영상을 녹화하지 못했습니다.");
    }
  }, [
    outputMode,
    runWebcamAnalysis,
    setErrorMessage,
    setStatus,
    webcam.stream,
  ]);

  const handleOutputModeChange = useCallback((nextOutputMode) => {
    if (interactionLocked) return;

    setOutputMode(nextOutputMode);
    resetAnalysis(
      inputMode === INPUT_MODES.WEBCAM && webcamConnected
        ? ANALYSIS_STATUS.CAMERA_READY
        : ANALYSIS_STATUS.IDLE,
    );
  }, [inputMode, interactionLocked, resetAnalysis, webcamConnected]);

  return (
    <main className="app-shell">
      <AppHeader
        inputMode={inputMode}
        onInputModeChange={handleInputModeChange}
        disabled={interactionLocked}
      />

      <div className="workspace-grid">
        <MediaWorkspace
          inputMode={inputMode}
          status={status}
          keypoints={displayedKeypoints}
          keypointsLiveReady={keypointLiveReady}
          outputMode={outputMode}
          onOutputModeChange={handleOutputModeChange}
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
          onWebcamStart={handleWebcamStart}
          isAnalyzing={isAnalyzing}
          controlsDisabled={interactionLocked}
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
