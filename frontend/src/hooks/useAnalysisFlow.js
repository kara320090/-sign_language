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
  const [uploadKeypoints, setUploadKeypoints] = useState(EMPTY_KEYPOINTS);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const requestIdRef = useRef(0);

  const applyResult = useCallback((nextResult) => {
    setResult(nextResult);
    setErrorMessage("");
    setUploadKeypoints(nextResult.keypoints ?? EMPTY_KEYPOINTS);
    setStatus(nextResult.llmFallback ? ANALYSIS_STATUS.LLM_FALLBACK : ANALYSIS_STATUS.SUCCESS);
  }, []);

  const resetAnalysis = useCallback((nextStatus = ANALYSIS_STATUS.IDLE) => {
    requestIdRef.current += 1;
    setResult(null);
    setErrorMessage("");
    setUploadKeypoints(EMPTY_KEYPOINTS);
    setIsAnalyzing(false);
    setStatus(nextStatus);
  }, []);

  const runUploadAnalysis = useCallback(async (file, outputMode) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    if (!file) {
      setStatus(ANALYSIS_STATUS.ERROR);
      setErrorMessage("분석할 영상 파일을 먼저 선택해 주세요.");
      return;
    }

    setIsAnalyzing(true);
    setStatus(ANALYSIS_STATUS.LOADING);
    setErrorMessage("");

    try {
      const nextResult = await analyzeUpload({ file, outputMode });
      if (requestIdRef.current !== requestId) return;
      applyResult(nextResult);
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      window.alert("업로드 실패");
      setResult(null);
      setStatus(ANALYSIS_STATUS.ERROR);
      setErrorMessage(error?.message ?? "업로드한 영상 분석에 실패했습니다.");
    } finally {
      if (requestIdRef.current === requestId) {
        setIsAnalyzing(false);
      }
    }
  }, [applyResult]);

  const runWebcamAnalysis = useCallback(async (outputMode) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    setIsAnalyzing(true);
    setStatus(ANALYSIS_STATUS.LOADING);
    setErrorMessage("");

    try {
      const nextResult = await analyzeWebcam({ outputMode, imageBlob });
      if (requestIdRef.current !== requestId) return;
      applyResult(nextResult);
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      setResult(null);
      setStatus(ANALYSIS_STATUS.ERROR);
      setErrorMessage(error?.message ?? "실시간 분석 결과를 가져올 수 없습니다.");
    } finally {
      if (requestIdRef.current === requestId) {
        setIsAnalyzing(false);
      }
    }
  }, [applyResult]);

  return {
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
  };
}
