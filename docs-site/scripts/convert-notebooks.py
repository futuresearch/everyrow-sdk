#!/usr/bin/env python3
"""Convert case study notebooks to HTML fragments for the docs site."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Use resolve() for absolute paths - works regardless of working directory
SCRIPT_DIR = Path(__file__).resolve().parent  # docs-site/scripts/
DOCS_SITE_DIR = SCRIPT_DIR.parent  # docs-site/
REPO_ROOT = DOCS_SITE_DIR.parent  # repo root

NOTEBOOKS_DIR = REPO_ROOT / "docs" / "case_studies"
OUTPUT_DIR = DOCS_SITE_DIR / "src" / "notebooks"


def convert(notebook: Path) -> str:
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--template",
            "basic",
            "--output-dir",
            str(OUTPUT_DIR),
            "--output",
            notebook.parent.name,
            str(notebook),
        ],
        capture_output=True,
        check=True,
    )
    return notebook.parent.name


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = list(NOTEBOOKS_DIR.glob("*/notebook.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(convert, notebooks))

    print(f"Converted {len(results)} notebooks")


if __name__ == "__main__":
    main()
