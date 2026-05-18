"""Install shared Python dependencies for word_AI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def install_requirements(requirements_path: Path, upgrade_pip: bool = True) -> None:
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    python_exe = sys.executable

    if upgrade_pip:
        run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

    run_command([python_exe, "-m", "pip", "install", "-r", str(requirements_path)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements",
        type=str,
        default=str(Path(__file__).resolve().parent / "requirements.txt"),
        help="Path to requirements.txt",
    )
    parser.add_argument(
        "--no-upgrade-pip",
        action="store_true",
        help="Skip pip upgrade step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements_path = Path(args.requirements)

    print("=" * 60)
    print("word_AI dependency installer (shared)")
    print("Python:", sys.executable)
    print("Requirements:", requirements_path)
    print("=" * 60)

    install_requirements(requirements_path, upgrade_pip=not args.no_upgrade_pip)

    print("\n[DONE] Dependency installation completed.")


if __name__ == "__main__":
    main()
