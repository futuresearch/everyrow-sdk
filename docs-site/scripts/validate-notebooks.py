#!/usr/bin/env python3
"""Validate that case study notebooks have proper title and description structure.

Each notebook must:
1. Start with a markdown cell
2. Have an H1 title (# Title) as the first line
3. Have a description in metadata.everyrow.description
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_SITE_DIR = SCRIPT_DIR.parent
REPO_ROOT = DOCS_SITE_DIR.parent
NOTEBOOKS_DIR = REPO_ROOT / "docs" / "case_studies"


def validate_notebook(notebook_path: Path) -> list[str]:
    """Validate a notebook's structure. Returns list of error messages."""
    errors = []
    slug = notebook_path.parent.name

    with open(notebook_path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not cells:
        errors.append(f"{slug}: Notebook has no cells")
        return errors

    # Check first cell is markdown
    first_cell = cells[0]
    if first_cell.get("cell_type") != "markdown":
        errors.append(
            f"{slug}: First cell must be markdown, got {first_cell.get('cell_type')}"
        )
        return errors

    # Get the source content
    source = first_cell.get("source", [])
    if isinstance(source, list):
        content = "".join(source)
    else:
        content = source

    lines = content.strip().split("\n")
    if not lines:
        errors.append(f"{slug}: First cell is empty")
        return errors

    # Check for H1 title
    first_line = lines[0].strip()
    if not first_line.startswith("# "):
        errors.append(
            f"{slug}: First line must be an H1 title (# Title), got: {first_line[:50]!r}"
        )
        return errors

    title = first_line[2:].strip()
    if not title:
        errors.append(f"{slug}: H1 title is empty")
        return errors

    # Check for description in metadata
    metadata = nb.get("metadata", {})
    everyrow_meta = metadata.get("everyrow", {})
    description = everyrow_meta.get("description", "")

    if not description:
        errors.append(f"{slug}: Missing metadata.everyrow.description")
    elif len(description) > 160:
        errors.append(
            f"{slug}: Description too long ({len(description)} chars). "
            f"Keep under 160 for SEO."
        )

    return errors


def main() -> int:
    notebooks = list(NOTEBOOKS_DIR.glob("*/notebook.ipynb"))

    if not notebooks:
        print("No notebooks found in", NOTEBOOKS_DIR)
        return 1

    all_errors = []
    for notebook in sorted(notebooks):
        errors = validate_notebook(notebook)
        all_errors.extend(errors)

    if all_errors:
        print("Notebook validation failed:\n")
        for error in all_errors:
            print(f"  - {error}")
        print(f"\n{len(all_errors)} error(s) in {len(notebooks)} notebooks")
        return 1

    print(f"All {len(notebooks)} notebooks have valid title/description structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
