from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "compneurovis"
DOCS_REF = ROOT / "docs" / "reference"
EXAMPLES = ROOT / "examples"
SKILLS = ROOT / "skills"
SKILL_KIND_LABELS = {
    "authoring": "Authoring",
    "coverage": "Coverage",
    "debug": "Debugging",
    "orchestration": "Orchestration",
    "meta": "Repo Maintenance",
}
SKILL_SURFACE_LABELS = {
    "backend": "Backend",
    "cross-cutting": "Cross-Cutting",
    "docs": "Docs",
    "examples": "Examples",
    "frontend": "Frontend",
    "repo-infra": "Repo Infrastructure",
}
SKILL_STAGE_LABELS = {
    "debug": "Debug",
    "explore": "Explore",
    "implement": "Implement",
    "release": "Release",
    "verify": "Verify",
}
SKILL_TRUST_LABELS = {
    "general": "General",
    "maintainer-only": "Maintainer Only",
    "proposal-only": "Proposal Only",
}


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    kind: str
    surface: str
    stage: str
    trust: str
    path: Path


def parse_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if not in_frontmatter or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def list_python_modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def public_api_names() -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        module = importlib.import_module("compneurovis")
        return list(getattr(module, "__all__", []))
    finally:
        sys.path.pop(0)


def skill_entries() -> list[SkillEntry]:
    entries: list[SkillEntry] = []
    for skill_md in sorted(SKILLS.glob("*/SKILL.md")):
        metadata = parse_frontmatter(skill_md)
        missing = [
            key
            for key in ("name", "description", "kind", "surface", "stage", "trust")
            if not metadata.get(key)
        ]
        if missing:
            raise SystemExit(
                f"{relative(skill_md)} is missing skill frontmatter fields: {', '.join(missing)}"
            )
        if metadata["kind"] not in SKILL_KIND_LABELS:
            raise SystemExit(f"{relative(skill_md)} has unsupported skill kind: {metadata['kind']}")
        if metadata["surface"] not in SKILL_SURFACE_LABELS:
            raise SystemExit(
                f"{relative(skill_md)} has unsupported skill surface: {metadata['surface']}"
            )
        if metadata["stage"] not in SKILL_STAGE_LABELS:
            raise SystemExit(
                f"{relative(skill_md)} has unsupported skill stage: {metadata['stage']}"
            )
        if metadata["trust"] not in SKILL_TRUST_LABELS:
            raise SystemExit(
                f"{relative(skill_md)} has unsupported skill trust: {metadata['trust']}"
            )
        entries.append(
            SkillEntry(
                name=metadata["name"],
                description=metadata["description"],
                kind=metadata["kind"],
                surface=metadata["surface"],
                stage=metadata["stage"],
                trust=metadata["trust"],
                path=skill_md,
            )
        )
    return entries


def append_group(lines: list[str], heading: str, entries: list[SkillEntry]) -> None:
    lines.extend([f"### {heading}", ""])
    for entry in entries:
        lines.append(f"- `{entry.name}`")
    lines.append("")


def build_repo_map() -> str:
    lines = [
        "---",
        "title: Repository Map",
        "summary: Generated map of the current package, examples, docs, and skills tree.",
        "---",
        "",
        "# Repository Map",
        "",
        "## Packages",
    ]
    for module in list_python_modules():
        lines.append(f"- `{relative(module)}`")
    lines.extend(["", "## Examples"])
    for example in sorted(EXAMPLES.rglob("*.py")):
        lines.append(f"- `{relative(example)}`")
    lines.extend(["", "## Skills"])
    for entry in skill_entries():
        lines.append(f"- `{entry.name}`: `{relative(entry.path)}`")
    return "\n".join(lines) + "\n"


def build_api_index() -> str:
    names = public_api_names()
    lines = [
        "---",
        "title: Public API Index",
        "summary: Generated list of names exported by compneurovis.__init__.",
        "---",
        "",
        "# Public API Index",
        "",
    ]
    for name in names:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def build_example_index() -> str:
    lines = [
        "---",
        "title: Example Index",
        "summary: Generated list of runnable examples grouped by directory.",
        "---",
        "",
        "# Example Index",
        "",
    ]
    for example in sorted(EXAMPLES.rglob("*.py")):
        lines.append(f"- `{relative(example)}`")
    return "\n".join(lines) + "\n"


def build_skill_index() -> str:
    entries = skill_entries()
    lines = [
        "---",
        "title: Skill Index",
        "summary: Generated taxonomy and catalog of repo-owned skills.",
        "---",
        "",
        "# Skill Index",
        "",
        "The canonical skill files stay under `skills/*/SKILL.md`. This generated index",
        "groups them by workflow metadata so discovery does not depend on a flat path list.",
        "",
        "## By Kind",
        "",
    ]
    for kind, heading in SKILL_KIND_LABELS.items():
        append_group(lines, heading, [entry for entry in entries if entry.kind == kind])

    lines.extend(["## By Surface", ""])
    for surface, heading in SKILL_SURFACE_LABELS.items():
        append_group(lines, heading, [entry for entry in entries if entry.surface == surface])

    lines.extend(["## By Workflow Stage", ""])
    for stage, heading in SKILL_STAGE_LABELS.items():
        append_group(lines, heading, [entry for entry in entries if entry.stage == stage])

    lines.extend(["## By Trust", ""])
    for trust, heading in SKILL_TRUST_LABELS.items():
        append_group(lines, heading, [entry for entry in entries if entry.trust == trust])

    lines.extend(["## Full Catalog", ""])
    for entry in entries:
        metadata = ", ".join(
            [
                f"kind: {entry.kind}",
                f"surface: {entry.surface}",
                f"stage: {entry.stage}",
                f"trust: {entry.trust}",
            ]
        )
        lines.append(
            f"- `{entry.name}` ({metadata}): {entry.description} (`{relative(entry.path)}`)"
        )
    return "\n".join(lines) + "\n"


def generate() -> None:
    write(DOCS_REF / "repo-map.md", build_repo_map())
    write(DOCS_REF / "api-index.md", build_api_index())
    write(DOCS_REF / "example-index.md", build_example_index())
    write(DOCS_REF / "skill-index.md", build_skill_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs/reference index files.")
    parser.add_argument("--check", action="store_true", help="Fail if generated files are out of date.")
    args = parser.parse_args()

    expected = {
        DOCS_REF / "repo-map.md": build_repo_map(),
        DOCS_REF / "api-index.md": build_api_index(),
        DOCS_REF / "example-index.md": build_example_index(),
        DOCS_REF / "skill-index.md": build_skill_index(),
    }
    if args.check:
        stale = [path for path, content in expected.items() if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit(f"Generated reference docs are stale: {', '.join(relative(path) for path in stale)}")
        return
    generate()


if __name__ == "__main__":
    main()
