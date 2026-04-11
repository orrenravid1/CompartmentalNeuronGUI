from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs" / "architecture" / "docs-vocabulary.json"
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", "site"}

Severity = Literal["error", "warning"]
MatchMode = Literal["substring", "word", "regex"]


@dataclass(frozen=True)
class VocabularyRule:
    name: str
    severity: Severity
    pattern: str
    match_mode: MatchMode
    paths: tuple[str, ...]
    exclude_paths: tuple[str, ...]
    ignore_code_blocks: bool
    ignore_inline_code: bool
    ignore_front_matter: bool
    message: str

    def matches(self, text: str) -> bool:
        if self.match_mode == "substring":
            return self.pattern in text
        if self.match_mode == "word":
            return re.search(rf"\b{re.escape(self.pattern)}\b", text) is not None
        return re.search(self.pattern, text) is not None


def load_rules(config_path: Path = CONFIG_PATH) -> list[VocabularyRule]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    rules: list[VocabularyRule] = []
    for entry in payload.get("rules", []):
        rules.append(
            VocabularyRule(
                name=entry["name"],
                severity=entry["severity"],
                pattern=entry["pattern"],
                match_mode=entry.get("match_mode", "substring"),
                paths=tuple(entry["paths"]),
                exclude_paths=tuple(entry.get("exclude_paths", ())),
                ignore_code_blocks=bool(entry.get("ignore_code_blocks", True)),
                ignore_inline_code=bool(entry.get("ignore_inline_code", True)),
                ignore_front_matter=bool(entry.get("ignore_front_matter", True)),
                message=entry.get("message", f"matched {entry['pattern']!r}"),
            )
        )
    return rules


def iter_markdown_files(root: Path, path_specs: tuple[str, ...], exclude_specs: tuple[str, ...]) -> list[Path]:
    exclude = {(root / spec).resolve() for spec in exclude_specs}
    files: list[Path] = []
    for spec in path_specs:
        path = (root / spec).resolve()
        if not path.exists() or path in exclude:
            continue
        if path.is_file():
            if path.suffix.lower() == ".md":
                files.append(path)
            continue
        for candidate in path.rglob("*.md"):
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            resolved = candidate.resolve()
            if resolved in exclude:
                continue
            files.append(resolved)
    return sorted(set(files))


def iter_scannable_lines(text: str, rule: VocabularyRule) -> list[tuple[int, str]]:
    lines = text.splitlines()
    results: list[tuple[int, str]] = []
    in_front_matter = False
    in_code_fence = False

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        if rule.ignore_front_matter and lineno == 1 and stripped == "---":
            in_front_matter = True
            continue
        if in_front_matter:
            if stripped == "---":
                in_front_matter = False
            continue

        if rule.ignore_code_blocks and stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence and rule.ignore_code_blocks:
            continue

        scan_line = INLINE_CODE_PATTERN.sub("", line) if rule.ignore_inline_code else line
        results.append((lineno, scan_line))
    return results


def scan_rule(root: Path, rule: VocabularyRule) -> list[str]:
    findings: list[str] = []
    for path in iter_markdown_files(root, rule.paths, rule.exclude_paths):
        text = path.read_text(encoding="utf-8")
        for lineno, scan_line in iter_scannable_lines(text, rule):
            if rule.matches(scan_line):
                rel = path.relative_to(root).as_posix()
                findings.append(
                    f"{rel}:{lineno}: {rule.severity} [{rule.name}] {rule.message}"
                )
    return findings


def scan_docs_vocabulary(root: Path = ROOT, config_path: Path = CONFIG_PATH) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for rule in load_rules(config_path):
        findings = scan_rule(root, rule)
        if rule.severity == "error":
            errors.extend(findings)
        else:
            warnings.extend(findings)
    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan markdown docs for hard vocabulary errors and optional softer wording drift."
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit non-zero when warning-level docs vocabulary findings are present.",
    )
    args = parser.parse_args()

    errors, warnings = scan_docs_vocabulary()
    for finding in errors:
        print(finding)
    for finding in warnings:
        print(finding)

    if errors or (args.fail_on_warnings and warnings):
        raise SystemExit(1)
    if warnings:
        print(f"Docs vocabulary OK with {len(warnings)} warning(s)")
    else:
        print("Docs vocabulary OK")


if __name__ == "__main__":
    main()
