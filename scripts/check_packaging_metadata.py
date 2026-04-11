from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
POETRY_LOCK = ROOT / "poetry.lock"


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main_dependencies(pyproject: dict[str, Any]) -> dict[str, Any]:
    dependencies = dict(pyproject["tool"]["poetry"]["dependencies"])
    dependencies.pop("python", None)
    return dependencies


def group_dependencies(pyproject: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = pyproject["tool"]["poetry"].get("group", {})
    resolved: dict[str, dict[str, Any]] = {}
    for group_name, payload in groups.items():
        resolved[group_name] = dict(payload.get("dependencies", {}))
    return resolved


def extras(pyproject: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: list(values)
        for name, values in dict(pyproject["tool"]["poetry"].get("extras", {})).items()
    }


def optional_dependency_names(dependencies: dict[str, Any]) -> set[str]:
    return {
        name
        for name, spec in dependencies.items()
        if isinstance(spec, dict) and spec.get("optional") is True
    }


def lock_package_names(lock: dict[str, Any]) -> set[str]:
    return {
        canonicalize_name(entry["name"])
        for entry in lock.get("package", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }


def normalize_extra_map(extra_map: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        name: sorted(canonicalize_name(value) for value in values)
        for name, values in sorted(extra_map.items())
    }


def validate_packaging_metadata(root: Path = ROOT) -> list[str]:
    pyproject = load_toml(root / "pyproject.toml")
    lock = load_toml(root / "poetry.lock")

    declared_main_dependencies = main_dependencies(pyproject)
    declared_group_dependencies = group_dependencies(pyproject)
    declared_extras = extras(pyproject)
    optional_dependencies = optional_dependency_names(declared_main_dependencies)
    dependency_name_map = {
        canonicalize_name(name): name for name in declared_main_dependencies
    }
    optional_dependency_names_canonical = {
        canonicalize_name(name) for name in optional_dependencies
    }
    locked_package_names = lock_package_names(lock)
    locked_extras = {
        name: list(values)
        for name, values in dict(lock.get("extras", {})).items()
    }

    errors: list[str] = []

    for extra_name, targets in sorted(declared_extras.items()):
        for target in targets:
            target_key = canonicalize_name(target)
            if target_key not in dependency_name_map:
                errors.append(
                    f"extra '{extra_name}' references unknown dependency '{target}' in [tool.poetry.dependencies]"
                )
                continue
            if target_key not in optional_dependency_names_canonical:
                errors.append(
                    f"extra '{extra_name}' references non-optional dependency '{target}'"
                )

    individually_installable = {
        canonicalize_name(target)
        for extra_name, targets in declared_extras.items()
        if extra_name != "all"
        for target in targets
    }
    missing_individual_extras = sorted(
        dependency_name_map[name]
        for name in optional_dependency_names_canonical - individually_installable
    )
    if missing_individual_extras:
        errors.append(
            "optional dependencies missing non-'all' extra exposure: "
            + ", ".join(missing_individual_extras)
        )

    expected_all = sorted(canonicalize_name(name) for name in optional_dependencies)
    actual_all = sorted(canonicalize_name(name) for name in declared_extras.get("all", []))
    if actual_all != expected_all:
        errors.append(
            "extra 'all' must contain every optional dependency exactly once: "
            f"expected {json.dumps(sorted(optional_dependencies))}, "
            f"found {json.dumps(declared_extras.get('all', []))}"
        )

    for dependency_name in sorted(declared_main_dependencies):
        if canonicalize_name(dependency_name) not in locked_package_names:
            errors.append(
                f"poetry.lock is missing main dependency '{dependency_name}' declared in pyproject.toml"
            )
    for group_name, dependencies in sorted(declared_group_dependencies.items()):
        for dependency_name in sorted(dependencies):
            if canonicalize_name(dependency_name) not in locked_package_names:
                errors.append(
                    f"poetry.lock is missing group dependency '{dependency_name}' from group '{group_name}'"
                )

    normalized_declared_extras = normalize_extra_map(declared_extras)
    normalized_locked_extras = normalize_extra_map(locked_extras)
    if normalized_locked_extras != normalized_declared_extras:
        errors.append(
            "poetry.lock extras do not match pyproject.toml extras: "
            f"expected {json.dumps(normalized_declared_extras, sort_keys=True)}, "
            f"found {json.dumps(normalized_locked_extras, sort_keys=True)}"
        )

    return errors


def main() -> None:
    errors = validate_packaging_metadata()
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("Packaging metadata OK")


if __name__ == "__main__":
    main()
