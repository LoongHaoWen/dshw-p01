from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "data/stock",
    "data/index",
    "data/macro",
    "data/finance",
    "data/clean",
    "data/combined",
    "output",
    "codes",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "download_log.txt",
    "01_download.ipynb",
    "02_clean.ipynb",
    "03_analysis.ipynb",
    "report.html",
]


def main() -> None:
    for relative_dir in REQUIRED_DIRS:
        os.makedirs(PROJECT_DIR / relative_dir, exist_ok=True)

    for relative_file in REQUIRED_FILES:
        path = PROJECT_DIR / relative_file
        if not path.exists():
            path.touch()

    print(f"Project structure checked under: {PROJECT_DIR}")


if __name__ == "__main__":
    main()
