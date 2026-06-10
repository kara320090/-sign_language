import { InputModeTabs } from "./InputModeTabs";

export function AppHeader({ inputMode, onInputModeChange, disabled }) {
  return (
    <header className="app-header">
      <nav className="main-navigation" aria-label="상단 네비게이션">
        <a className="nav-link active" href="#converter">수어 변환</a>
      </nav>
      <InputModeTabs
        value={inputMode}
        onChange={onInputModeChange}
        disabled={disabled}
      />
    </header>
  );
}
