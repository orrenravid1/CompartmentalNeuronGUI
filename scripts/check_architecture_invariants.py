from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs" / "architecture" / "invariants.json"
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache"}


@dataclass(frozen=True)
class BannedTermRule:
    term: str
    paths: tuple[str, ...]
    exclude_paths: tuple[str, ...]


def load_rules() -> tuple[list[BannedTermRule], list[list[str]]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rules = [
        BannedTermRule(
            term=entry["term"],
            paths=tuple(entry["paths"]),
            exclude_paths=tuple(entry.get("exclude_paths", ())),
        )
        for entry in payload.get("banned_terms", [])
    ]
    commands = [list(command) for command in payload.get("commands", [])]
    return rules, commands


def iter_files(path_specs: tuple[str, ...], exclude_specs: tuple[str, ...]) -> list[Path]:
    exclude = {(ROOT / spec).resolve() for spec in exclude_specs}
    files: list[Path] = []
    for spec in path_specs:
        path = (ROOT / spec).resolve()
        if not path.exists() or path in exclude:
            continue
        if path.is_file():
            files.append(path)
            continue
        for candidate in path.rglob("*"):
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            if not candidate.is_file():
                continue
            if candidate.resolve() in exclude:
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files.append(candidate.resolve())
    return sorted(set(files))


def scan_banned_terms(rules: list[BannedTermRule]) -> list[str]:
    violations: list[str] = []
    for rule in rules:
        for path in iter_files(rule.paths, rule.exclude_paths):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if rule.term in line:
                    rel = path.relative_to(ROOT).as_posix()
                    violations.append(f"{rel}:{lineno}: banned term '{rule.term}'")
    return violations


def run_commands(commands: list[list[str]]) -> list[str]:
    failures: list[str] = []
    for command in commands:
        actual = [sys.executable if part == "python" else part for part in command]
        completed = subprocess.run(
            actual,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            failures.append(
                "\n".join(
                    [
                        f"command failed: {' '.join(command)}",
                        completed.stdout.rstrip(),
                        completed.stderr.rstrip(),
                    ]
                ).strip()
            )
    return failures


def main() -> None:
    rules, commands = load_rules()
    violations = scan_banned_terms(rules)
    failures = run_commands(commands)
    if violations or failures:
        for violation in violations:
            print(violation)
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("Architecture invariants OK")


if __name__ == "__main__":
    main()
