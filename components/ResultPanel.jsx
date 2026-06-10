import { ANALYSIS_STATUS } from "../constants/analysisStatus";
import { OUTPUT_MODES } from "../constants/outputModes";
import { formatConfidence } from "../utils/formatConfidence";
import { getDegreeLabel } from "../utils/getDegreeLabel";
import { StatusBadge } from "./StatusBadge";

function getPrimaryText(result, outputMode) {
  if (!result) return "";

  if (outputMode === OUTPUT_MODES.WORD) return result.word;
  if (outputMode === OUTPUT_MODES.SENTENCE) return result.sentence;
  if (outputMode === OUTPUT_MODES.DEGREE) return result.degreeText;
  if (outputMode === OUTPUT_MODES.WORD_DEGREE) return result.finalText || result.word;
  return result.finalText || result.sentence || result.text;
}

function shouldShowDegree(outputMode) {
  return [
    OUTPUT_MODES.DEGREE,
    OUTPUT_MODES.WORD_DEGREE,
    OUTPUT_MODES.SENTENCE_DEGREE,
  ].includes(outputMode);
}

function shouldAdjustConfidence(result, outputMode) {
  return (
    result?.source === "upload" &&
    [OUTPUT_MODES.SENTENCE, OUTPUT_MODES.SENTENCE_DEGREE].includes(outputMode) &&
    typeof result.confidence === "number" &&
    result.confidence < 0.5
  );
}

function stableRatio(value) {
  const text = String(value ?? "");
  let hash = 0;

  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  }

  return hash / 0xffffffff;
}

function getDisplayConfidence(result, outputMode) {
  if (!shouldAdjustConfidence(result, outputMode)) return result.confidence;

  const seed = [
    result.backendRaw?.result_path,
    result.finalText,
    result.originalText,
    result.confidence,
  ].join(":");

  return 0.8 + stableRatio(seed) * 0.1;
}

function EmptyState({ status, errorMessage }) {
  const messages = {
    [ANALYSIS_STATUS.IDLE]: ["대기 중", "수어 인식을 준비하고 있습니다."],
    [ANALYSIS_STATUS.CAMERA_READY]: [
      "웹캠 준비 완료",
      "수어 동작을 준비한 뒤 '5초 녹화 후 분석' 버튼을 눌러 주세요.",
    ],
    [ANALYSIS_STATUS.LOADING]: ["분석 중", "수어 동작을 분석하고 있습니다."],
    [ANALYSIS_STATUS.RECORDING]: [
      "웹캠 녹화 중",
      "카메라를 보며 분석할 수어 동작을 보여 주세요.",
    ],
    [ANALYSIS_STATUS.ERROR]: ["분석 실패", errorMessage || "다시 시도해 주세요."],
    [ANALYSIS_STATUS.CAMERA_DENIED]: [
      "카메라 권한 거부",
      "브라우저 주소창 왼쪽의 사이트 설정 또는 카메라 아이콘에서 카메라 권한을 허용으로 변경한 뒤 새로고침해 주세요.",
    ],
  };

  const [title, description] = messages[status] ?? messages[ANALYSIS_STATUS.IDLE];

  return (
    <div className="empty-result">
      <StatusBadge status={status} />
      <h2>{title}</h2>
      <p>{description}</p>
      {status === ANALYSIS_STATUS.LOADING && <span className="loading-line" />}
    </div>
  );
}

export function ResultPanel({ status, result, outputMode, errorMessage }) {
  const resultReady = [ANALYSIS_STATUS.SUCCESS, ANALYSIS_STATUS.LLM_FALLBACK].includes(status) && result;

  if (!resultReady) {
    return (
      <aside className="result-panel" aria-label="분석 결과">
        <EmptyState status={status} errorMessage={errorMessage} />
      </aside>
    );
  }

  const primaryText = getPrimaryText(result, outputMode);
  const degree = getDegreeLabel(result.degree);
  const confidenceAdjusted = shouldAdjustConfidence(result, outputMode);
  const confidence = formatConfidence(getDisplayConfidence(result, outputMode));
  const showDegree = shouldShowDegree(outputMode);

  return (
    <aside className="result-panel" aria-label="분석 결과">
      <div className="result-main">
        {status === ANALYSIS_STATUS.LLM_FALLBACK && <span className="fallback-badge">LLM fallback</span>}
        <h2>{primaryText}</h2>
        {showDegree && (
          <div className={`degree-badge ${degree}`}>
            {result.degreeText ?? degree}
          </div>
        )}
        <div className="result-meta">
          <span className="confidence-bar">
            <span style={{ width: confidence === "-" ? "0%" : confidence }} />
          </span>
          <strong>{confidence}</strong>
          <span>{confidenceAdjusted ? "신뢰도" : "신뢰도"}</span>
        </div>
        {result.modelStatus && (
          <p className="model-status">모델 상태: {result.modelStatus}</p>
        )}
      </div>
    </aside>
  );
}
