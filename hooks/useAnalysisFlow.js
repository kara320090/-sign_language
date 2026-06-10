import { useCallback, useRef, useState } from "react";
import { analyzeUpload, analyzeWebcam } from "../api/analysisApi";
import { ANALYSIS_STATUS } from "../constants/analysisStatus";


const EMPTY_KEYPOINTS = {
  hands: false,
  face: false,
  pose: false,
};


export function useAnalysisFlow() {
  const [status, setStatus] = useState(ANALYSIS_STATUS.IDLE);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [analysisKeypoints, setAnalysisKeypoints] = useState(EMPTY_KEYPOINTS);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const requestIdRef = useRef(0);
  const abortControllerRef = useRef(null);

  const applyResult = useCallback((nextResult) => {
    setResult(nextResult);
    setErrorMessage("");
    setAnalysisKeypoints(nextResult.keypoints ?? EMPTY_KEYPOINTS);
    setStatus(
      nextResult.llmFallback
        ? ANALYSIS_STATUS.LLM_FALLBACK
        : ANALYSIS_STATUS.SUCCESS,
    );
  }, []);

  const applyError = useCallback((error, fallbackMessage) => {
    setResult(null);
    setAnalysisKeypoints(EMPTY_KEYPOINTS);
    setStatus(ANALYSIS_STATUS.ERROR);
    setErrorMessage(error?.message ?? fallbackMessage);
  }, []);

  const resetAnalysis = useCallback((nextStatus = ANALYSIS_STATUS.IDLE) => {
    requestIdRef.current += 1;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setResult(null);
    setErrorMessage("");
    setAnalysisKeypoints(EMPTY_KEYPOINTS);
    setIsAnalyzing(false);
    setStatus(nextStatus);
  }, []);

  const runUploadAnalysis = useCallback(async (file, outputMode) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    if (!file) {
      applyError(null, "분석할 영상 파일을 먼저 선택해 주세요.");
      return;
    }

    setIsAnalyzing(true);
    setStatus(ANALYSIS_STATUS.LOADING);
    setErrorMessage("");
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const nextResult = await analyzeUpload({
        file,
        outputMode,
        signal: controller.signal,
      });
      if (requestIdRef.current !== requestId) return;
      applyResult(nextResult);
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      applyError(error, "업로드한 영상 분석에 실패했습니다.");
    } finally {
      if (requestIdRef.current === requestId) {
        abortControllerRef.current = null;
        setIsAnalyzing(false);
      }
    }
  }, [applyError, applyResult]);

  const runWebcamAnalysis = useCallback(async (videoBlob, outputMode) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    setIsAnalyzing(true);
    setStatus(ANALYSIS_STATUS.LOADING);
    setErrorMessage("");
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const nextResult = await analyzeWebcam({
        videoBlob,
        outputMode,
        signal: controller.signal,
      });
      if (requestIdRef.current !== requestId) return;
      applyResult(nextResult);
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      applyError(error, "웹캠 영상 분석에 실패했습니다.");
    } finally {
      if (requestIdRef.current === requestId) {
        abortControllerRef.current = null;
        setIsAnalyzing(false);
      }
    }
  }, [applyError, applyResult]);

  return {
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
  };
}
