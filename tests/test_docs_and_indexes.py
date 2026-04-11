from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_MARKDOWN_FILES = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
)
CONTRIB_INSTALL_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "getting-started.md",
)
FORBIDDEN_BARE_MKDOCS_COMMAND_PATTERNS = (
    re.compile(r"(?<!python -m )mkdocs serve\b"),
    re.compile(r"(?<!python -m )mkdocs build --strict\b"),
)
README_PUBLIC_API_SECTION_PATTERN = re.compile(r"^## Public API\n(?P<body>.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
README_PUBLIC_API_REQUIRED_NAMES = (
    "NeuronSession",
    "JaxleySession",
    "build_neuron_app",
    "build_jaxley_app",
    "build_surface_app",
    "build_replay_app",
    "run_app",
)
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+)\)")
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
ROOT_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.:-])"
    r"(?P<path>"
    r"AGENTS\.md|README\.md|LICENSE|pyproject\.toml|poetry\.lock|"
    r"(?:\.compneurovis|\.github|docs|skills|scripts|src|tests|examples|res)(?:/[A-Za-z0-9._-]+)*/?"
    r")"
    r"(?![A-Za-z0-9_.-])"
)
ABSOLUTE_LOCAL_PATH_PATTERN = re.compile(r"^(?:/[A-Za-z]:/|[A-Za-z]:[/\\])")


def markdown_files() -> list[Path]:
    files = [path for path in ROOT_MARKDOWN_FILES if path.exists()]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    files.extend(sorted((ROOT / "skills").rglob("SKILL.md")))
    if (ROOT / ".github").exists():
        files.extend(sorted((ROOT / ".github").rglob("*.md")))
    if (ROOT / ".compneurovis").exists():
        files.extend(sorted((ROOT / ".compneurovis").rglob("*.md")))
    return files


def is_within_root(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT or ROOT in resolved.parents


def normalize_markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target and not target.startswith(("http://", "https://", "mailto:")):
        target = target.split(" ", 1)[0]
    return target


def resolve_markdown_target(path: Path, target: str) -> str | None:
    normalized = normalize_markdown_target(target)
    if not normalized or normalized.startswith("#"):
        return None
    if normalized.startswith(("http://", "https://", "mailto:")):
        return None
    if ABSOLUTE_LOCAL_PATH_PATTERN.match(normalized):
        return f"{path.relative_to(ROOT).as_posix()}: absolute local markdown path is not allowed: {normalized}"
    if normalized.startswith("/"):
        return f"{path.relative_to(ROOT).as_posix()}: root-absolute markdown path is not allowed: {normalized}"

    target_path = normalized.split("#", 1)[0]
    candidate = (path.parent / target_path).resolve()
    if not is_within_root(candidate):
        return f"{path.relative_to(ROOT).as_posix()}: markdown link escapes repo root: {normalized}"
    if not candidate.exists():
        return f"{path.relative_to(ROOT).as_posix()}: unresolved markdown link: {normalized}"
    return None


def resolve_root_relative_path(path: Path, candidate: str) -> str | None:
    resolved = (ROOT / candidate.rstrip("/")).resolve()
    if not is_within_root(resolved):
        return f"{path.relative_to(ROOT).as_posix()}: path escapes repo root: {candidate}"
    if not resolved.exists():
        return f"{path.relative_to(ROOT).as_posix()}: unresolved repo path: {candidate}"
    return None


def iter_root_relative_candidates(text: str) -> list[str]:
    return [match.group("path") for match in ROOT_PATH_PATTERN.finditer(text)]


def test_generated_indexes_are_in_sync():
    subprocess.run(
        [sys.executable, "scripts/generate_indexes.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_architecture_invariants_hold():
    subprocess.run(
        [sys.executable, "scripts/check_architecture_invariants.py"],
        cwd=ROOT,
        check=True,
    )


def test_mkdocs_builds_in_strict_mode():
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=ROOT,
        env=os.environ | {"NO_MKDOCS_2_WARNING": "true"},
        check=True,
    )


def test_docs_and_skills_have_front_matter():
    files = list((ROOT / "docs").rglob("*.md")) + list((ROOT / "skills").rglob("SKILL.md"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"missing front matter start: {path}"
        assert "\n---\n" in text[4:], f"missing front matter end: {path}"


def test_markdown_paths_resolve():
    unresolved: list[str] = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            error = resolve_markdown_target(path, match.group("target"))
            if error:
                unresolved.append(error)

        fenced_blocks = FENCED_CODE_BLOCK_PATTERN.findall(text)
        text_without_fences = FENCED_CODE_BLOCK_PATTERN.sub("", text)

        candidate_texts = list(fenced_blocks)
        candidate_texts.extend(INLINE_CODE_PATTERN.findall(text_without_fences))

        for candidate_text in candidate_texts:
            for candidate in iter_root_relative_candidates(candidate_text):
                error = resolve_root_relative_path(path, candidate)
                if error:
                    unresolved.append(error)
    assert not unresolved, "unresolved markdown paths:\n" + "\n".join(sorted(set(unresolved)))


def test_readme_public_api_mentions_backend_and_builder_entrypoints():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = README_PUBLIC_API_SECTION_PATTERN.search(text)
    assert match is not None, "README.md is missing a Public API section"
    public_api_section = match.group("body")
    missing = [name for name in README_PUBLIC_API_REQUIRED_NAMES if name not in public_api_section]
    assert not missing, "README.md Public API section is missing: " + ", ".join(missing)


def test_docs_use_python_module_invocation_for_mkdocs_commands():
    violations: list[str] = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_BARE_MKDOCS_COMMAND_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()}:{line}: use 'python -m {match.group(0)}'"
                )
    assert not violations, "bare mkdocs commands found:\n" + "\n".join(violations)


def test_contributor_docs_use_named_contrib_extra():
    required_command = 'pip install -e ".[contrib]"'
    legacy_fragment = 'pip install -e . pytest mkdocs mkdocs-material "mkdocstrings[python]"'

    missing: list[str] = []
    legacy_hits: list[str] = []
    for path in CONTRIB_INSTALL_DOCS:
        text = path.read_text(encoding="utf-8")
        if required_command not in text:
            missing.append(path.relative_to(ROOT).as_posix())
        if legacy_fragment in text:
            legacy_hits.append(path.relative_to(ROOT).as_posix())

    assert not missing, "docs missing contributor extra install command:\n" + "\n".join(missing)
    assert not legacy_hits, "docs still use legacy raw contributor dependency command:\n" + "\n".join(legacy_hits)
