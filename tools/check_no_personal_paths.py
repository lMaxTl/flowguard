#!/usr/bin/env python3
"""Refuse to commit files containing absolute or personal filesystem paths.

Committed notebook outputs in this project's pre-release history leaked strings
like ``/Users/<realname>/Documents/...``. That is both a privacy problem and a
reproducibility problem: a path that only exists on one laptop is not a path
anyone else can follow.

Used as a pre-commit hook (see ``.pre-commit-config.yaml``) and runnable
directly::

    python tools/check_no_personal_paths.py $(git ls-files)
"""

from __future__ import annotations

import re
import sys

PATTERNS: list[tuple[str, re.Pattern[bytes]]] = [
    ("macOS home directory", re.compile(rb"/Users/[A-Za-z0-9._-]+")),
    ("Linux home directory", re.compile(rb"/home/(?!public\b)[A-Za-z0-9._-]+/")),
    ("Windows user directory", re.compile(rb"[A-Za-z]:\\+Users\\+[A-Za-z0-9._-]+")),
    ("Windows absolute path", re.compile(rb"[A-Za-z]:\\+[A-Za-z0-9 _-]+\\+[A-Za-z0-9 _-]+\\+")),
]

# Vendored upstream code is out of scope: we keep it byte-close to the original.
SKIP_PREFIXES = ("defenses/", "pretrainedmodels/")


def main(paths: list[str]) -> int:
    findings: list[str] = []

    for path in paths:
        if path.replace("\\", "/").startswith(SKIP_PREFIXES):
            continue
        try:
            with open(path, "rb") as handle:
                blob = handle.read()
        except (OSError, IsADirectoryError):
            continue

        for label, pattern in PATTERNS:
            match = pattern.search(blob)
            if match is None:
                continue
            line = blob[: match.start()].count(b"\n") + 1
            snippet = match.group(0).decode("utf-8", "replace")
            findings.append(f"  {path}:{line}: {label}: {snippet}")
            break

    if not findings:
        return 0

    print("Personal or absolute filesystem paths must not be committed:", file=sys.stderr)
    print("\n".join(findings), file=sys.stderr)
    print(
        "\nUse a path relative to the repository root, or read the location from a "
        "CLI flag / environment variable. For notebooks, run `nbstripout` on the file.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
