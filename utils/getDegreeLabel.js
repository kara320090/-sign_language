const DEGREE_LABELS = {
  weak: "weak",
  normal: "normal",
  strong: "strong",
};

export function getDegreeLabel(degree) {
  return DEGREE_LABELS[degree] ?? "normal";
}
