import { OUTPUT_MODE_OPTIONS } from "../constants/outputModes";

export function OutputModeSelector({ value, onChange }) {
  return (
    <div className="output-selector" aria-label="출력 모델 선택">
      {OUTPUT_MODE_OPTIONS.map((option) => (
        <button
          className={option.id === value ? "output-option active" : "output-option"}
          key={option.id}
          type="button"
          onClick={() => onChange(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
