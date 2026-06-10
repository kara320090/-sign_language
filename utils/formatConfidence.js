export function formatConfidence(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }

  return `${Math.round(value * 100)}%`;
}
