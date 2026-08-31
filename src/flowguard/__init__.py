from __future__ import annotations

import sys
from pathlib import Path


def _ensure_legacy_project_root_on_path() -> None:
    package_root = Path(__file__).resolve().parents[2]
    project_root = str(package_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_legacy_project_root_on_path()

__all__: list[str] = []
