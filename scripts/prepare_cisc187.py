#!/usr/bin/env python3
"""
prepare_cisc187.py — Convert the cisc187-reader-master repo into a JSONL
corpus suitable for LLaMA-Factory continued pretraining (stage: pt).

Output format: one JSON object per line, each with a "text" field.
LLaMA-Factory's dataset_info.json entry maps "prompt" -> "text" for the
pretraining stage.

Usage (run from the directory containing the cisc187-reader-master folder):
    python prepare_cisc187.py \\
        --repo ./cisc187-reader-master \\
        --out  ./cisc187_pt.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


# File extensions worth keeping for next-token training.
# .rst is the bulk of the textbook prose (Sphinx source).
# .txt is used for embedded code samples and quiz items.
# .py / .cpp / .h are scattered helper / example code.
INCLUDE_EXTS = {
    ".rst", ".txt", ".md",
    ".py",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".cmake", ".yml", ".yaml", ".json",
    ".sh",
}

# Skip generated output, build dirs, VCS metadata, etc.
SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn",
    "build", "dist", "node_modules", "vendor",
    "__pycache__", ".vscode", ".idea",
    "docs",        # rendered HTML — duplicates _sources content
    "_static", "_images", "_extensions",
    "doctrees", "plot_directive",
}

# Skip individual files that pollute training (license boilerplate, build state).
SKIP_FILE_NAMES = {
    "LICENSE.txt", "OpenDSA-license.txt",
    "build_info", "sphinx_settings.json", "sphinx-enki-info.txt",
}

# Files smaller than this many non-whitespace chars are dropped as noise.
MIN_LENGTH = 32


def collect_records(repo: Path):
    records = []
    skipped_bin = 0
    skipped_empty = 0

    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        if path.suffix.lower() not in INCLUDE_EXTS:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped_bin += 1
            continue

        stripped = text.strip()
        if len(stripped) < MIN_LENGTH:
            skipped_empty += 1
            continue

        # Prefix with the relative path so the model has a weak signal of
        # "which part of the textbook this chunk came from."
        rel = path.relative_to(repo)
        header = f"# File: {rel}\n\n"
        records.append({"text": header + stripped})

    return records, skipped_bin, skipped_empty


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True,
                    help="Path to cisc187-reader-master (or any source repo).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path.")
    args = ap.parse_args()

    if not args.repo.is_dir():
        print(f"error: --repo {args.repo} is not a directory", file=sys.stderr)
        sys.exit(1)

    records, skipped_bin, skipped_empty = collect_records(args.repo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {args.out}")
    print(f"  skipped {skipped_bin} non-utf8 file(s)")
    print(f"  skipped {skipped_empty} empty/too-short file(s)")


if __name__ == "__main__":
    main()
