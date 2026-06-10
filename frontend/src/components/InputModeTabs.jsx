import { INPUT_MODE_OPTIONS } from "../constants/inputModes";

export function InputModeTabs({ value, onChange }) {
  return (
    <div className="input-mode-tabs" aria-label="입력 방식 선택">
      {INPUT_MODE_OPTIONS.map((option) => (
        <button
          className={option.id === value ? "mode-tab active" : "mode-tab"}
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
