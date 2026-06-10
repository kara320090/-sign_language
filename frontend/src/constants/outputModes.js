export const OUTPUT_MODES = {
  WORD: "word",
  SENTENCE: "sentence",
  DEGREE: "degree",
  WORD_DEGREE: "word_degree",
  SENTENCE_DEGREE: "sentence_degree",
};

export const OUTPUT_MODE_OPTIONS = [
  { id: OUTPUT_MODES.WORD, label: "단어" },
  { id: OUTPUT_MODES.SENTENCE, label: "문장" },
  { id: OUTPUT_MODES.DEGREE, label: "표현정도" },
  { id: OUTPUT_MODES.WORD_DEGREE, label: "단어+표현" },
  { id: OUTPUT_MODES.SENTENCE_DEGREE, label: "문장+표현" },
];
