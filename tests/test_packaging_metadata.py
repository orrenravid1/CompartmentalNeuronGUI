from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "check_packaging_metadata.py"
SPEC = importlib.util.spec_from_file_location("check_packaging_metadata_script", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
check_packaging_metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_packaging_metadata)


def write_repo_metadata(tmp_path: Path, *, pyproject: str, lock: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (repo / "poetry.lock").write_text(lock, encoding="utf-8")
    return repo


def test_packaging_metadata_holds_for_repo():
    assert check_packaging_metadata.validate_packaging_metadata(ROOT) == []


def test_packaging_metadata_rejects_unknown_extra_targets(tmp_path: Path):
    repo = write_repo_metadata(
        tmp_path,
        pyproject="""
[tool.poetry]
name = "demo"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
pymoose = { version = "^4.1", optional = true }

[tool.poetry.extras]
moose = ["moose"]
all = ["moose"]
""",
        lock="""
version = "2.1"

[[package]]
name = "pymoose"
version = "4.1.2"

[extras]
moose = ["moose"]
all = ["moose"]
""",
    )

    errors = check_packaging_metadata.validate_packaging_metadata(repo)

    assert any("extra 'moose' references unknown dependency 'moose'" in error for error in errors)
    assert any("extra 'all' references unknown dependency 'moose'" in error for error in errors)


def test_packaging_metadata_rejects_missing_locked_group_dependency(tmp_path: Path):
    repo = write_repo_metadata(
        tmp_path,
        pyproject="""
[tool.poetry]
name = "demo"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
numpy = "^1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
mkdocs = "^1.6"
""",
        lock="""
version = "2.1"

[[package]]
name = "numpy"
version = "1.26.0"

[[package]]
name = "pytest"
version = "8.0.0"
""",
    )

    errors = check_packaging_metadata.validate_packaging_metadata(repo)

    assert "poetry.lock is missing group dependency 'mkdocs' from group 'dev'" in errors
