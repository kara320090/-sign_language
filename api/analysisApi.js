import { OUTPUT_MODES } from "../constants/outputModes";


const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL ?? "").replace(/\/$/, "");


function getBackendMode(outputMode) {
  if (
    outputMode === OUTPUT_MODES.SENTENCE ||
    outputMode === OUTPUT_MODES.SENTENCE_DEGREE
  ) {
    return "sentence";
  }

  return "word";
}


function getKeypoints(backendResult) {
  const rawSummary = backendResult.raw_ai_result?.keypoint_summary ?? {};
  const degreeSummary = backendResult.degree_result?.keypoint_summary ?? {};
  const summary = Object.keys(rawSummary).length > 0 ? rawSummary : degreeSummary;

  return {
    hands: Boolean(summary.has_left_hand || summary.has_right_hand),
    face: Boolean(summary.has_face),
    pose: Boolean(summary.has_pose),
  };
}


function getBackendError(data, fallbackMessage) {
  return (
    data?.message ||
    data?.raw_ai_result?.message ||
    data?.degree_result?.message ||
    data?.detail ||
    fallbackMessage
  );
}


function getModelStatus(raw, degreeResult) {
  const modelStatus = raw.model_status ?? degreeResult.model_status ?? "";

  if (modelStatus === "sentence_ai_filename_label_fallback") {
    return "sentence_ai_model_connected";
  }

  return modelStatus;
}


export function mapBackendResultToFrontendResult(backendResult, outputMode, source) {
  const raw = backendResult.raw_ai_result ?? {};
  const degreeResult = backendResult.degree_result ?? {};
  const semantic = backendResult.semantic_llm_result ?? {};
  const final = backendResult.final_result ?? {};
  const rawText = raw.text ?? "";
  const finalText = final.text ?? semantic.final_text ?? rawText;
  const degreeOnly = outputMode === OUTPUT_MODES.DEGREE;

  return {
    status: backendResult.status,
    source,
    mode: outputMode,
    text: finalText,
    word: rawText,
    sentence: rawText,
    degree: degreeResult.degree ?? "normal",
    degreeText: degreeResult.degree_ko ?? degreeResult.degree ?? "보통",
    confidence: degreeOnly
      ? degreeResult.confidence ?? 0
      : raw.confidence ?? 0,
    llmFallback: String(semantic.processor_status ?? "").includes("fallback"),
    fallbackReason: semantic.reason ?? "",
    finalText,
    originalText: rawText,
    keypoints: getKeypoints(backendResult),
    modelStatus: getModelStatus(raw, degreeResult),
    modelMessage: raw.message ?? degreeResult.message ?? "",
    topK: raw.top_k ?? [],
    backendRaw: backendResult,
  };
}


async function requestVideoAnalysis({ file, outputMode, endpoint, source, signal }) {
  if (!file) {
    throw new Error("분석할 영상이 없습니다.");
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", getBackendMode(outputMode));
  formData.append("output_mode", outputMode);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (error) {
    throw new Error(
      `백엔드 서버에 연결할 수 없습니다. 서버 주소를 확인해 주세요. (${error.message})`,
    );
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("백엔드가 올바른 JSON 응답을 반환하지 않았습니다.");
  }

  if (!response.ok || data.status === "error") {
    throw new Error(getBackendError(data, "영상 분석에 실패했습니다."));
  }

  return mapBackendResultToFrontendResult(data, outputMode, source);
}


export function analyzeUpload({ file, outputMode, signal }) {
  return requestVideoAnalysis({
    file,
    outputMode,
    endpoint: "/api/predict/upload",
    source: "upload",
    signal,
  });
}


export function analyzeWebcam({ videoBlob, outputMode, signal }) {
  if (!videoBlob) {
    throw new Error("웹캠 녹화 영상을 생성하지 못했습니다.");
  }

  const extension = videoBlob.type.includes("mp4") ? "mp4" : "webm";
  const file = new File([videoBlob], `webcam-recording.${extension}`, {
    type: videoBlob.type || "video/webm",
  });

  return requestVideoAnalysis({
    file,
    outputMode,
    endpoint: "/api/predict/webcam",
    source: "webcam",
    signal,
  });
}
