from __future__ import annotations

import argparse
import ast
import importlib
import re
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
SKILL_TAXONOMY_KEYS = ("kind", "surface", "stage", "trust")
EXAMPLE_GROUPS = (
    ("Live Simulation Backends", ("NEURON", "Jaxley")),
    ("Field and Surface Workflows", ("Static / Interactive", "Live", "Replay")),
    ("Debug and Architecture Probes", ("Debug",)),
)
TITLE_SPLIT_PATTERN = re.compile(r"\s[—–-]\s", re.UNICODE)


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    kind: str
    surface: str
    stage: str
    trust: str
    path: Path


@dataclass(frozen=True)
class ExampleEntry:
    title: str
    summary: str
    group: str
    subgroup: str
    path: Path


def parse_frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(?P<body>.*?)\n---(?:\n|$)", text, re.DOTALL)
    if match is None:
        return {}

    metadata: dict[str, object] = {}
    current_parent: str | None = None
    for raw_line in match.group("body").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            current_parent = None
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip()
            if value:
                metadata[key.strip()] = value
            else:
                metadata[key.strip()] = {}
                current_parent = key.strip()
            continue

        if current_parent is None or ":" not in stripped:
            continue
        container = metadata.get(current_parent)
        if not isinstance(container, dict):
            container = {}
            metadata[current_parent] = container
        key, value = stripped.split(":", 1)
        container[key.strip()] = value.strip()
    return metadata


def skill_frontmatter_value(frontmatter: dict[str, object], key: str) -> str:
    direct = frontmatter.get(key)
    if isinstance(direct, str) and direct:
        return direct
    nested = frontmatter.get("metadata")
    if isinstance(nested, dict):
        nested_value = nested.get(key)
        if isinstance(nested_value, str) and nested_value:
            return nested_value
    return ""


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


def humanize_identifier(value: str) -> str:
    words = value.replace("_", " ").split()
    normalized: list[str] = []
    for word in words:
        if word.lower() == "3d":
            normalized.append("3D")
        elif word.isupper():
            normalized.append(word)
        else:
            normalized.append(word.capitalize())
    return " ".join(normalized)


def parse_example_docstring(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return ast.get_docstring(module) or ""


def first_docstring_paragraph(docstring: str) -> str:
    lines: list[str] = []
    for raw_line in docstring.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines:
                break
            continue
        lines.append(stripped)
    return " ".join(lines)


def split_title_summary(first_line: str) -> tuple[str, str]:
    line = " ".join(first_line.split())
    if not line:
        return "", ""
    match = TITLE_SPLIT_PATTERN.search(line)
    if match is None:
        return line.rstrip("."), ""
    title = line[: match.start()].strip()
    summary = line[match.end() :].strip().rstrip(".")
    return title, summary


def infer_example_group(path: Path) -> tuple[str, str]:
    folder = path.parent.name
    stem = path.stem
    if folder == "neuron":
        return "Live Simulation Backends", "NEURON"
    if folder == "jaxley":
        return "Live Simulation Backends", "Jaxley"
    if folder == "surface_plot":
        if "replay" in stem:
            return "Field and Surface Workflows", "Replay"
        if "live" in stem:
            return "Field and Surface Workflows", "Live"
        return "Field and Surface Workflows", "Static / Interactive"
    return "Debug and Architecture Probes", "Debug"


def default_example_summary(path: Path) -> str:
    folder = path.parent.name
    if folder == "debug":
        return "Debug-oriented example for probing layout, session, or rendering behavior"
    if folder in {"neuron", "jaxley"}:
        return "Runnable live session example"
    return "Runnable field or surface visualization example"


def example_entries() -> list[ExampleEntry]:
    entries: list[ExampleEntry] = []
    for example in sorted(EXAMPLES.rglob("*.py")):
        docstring = parse_example_docstring(example)
        lead_paragraph = first_docstring_paragraph(docstring)
        title, summary = split_title_summary(lead_paragraph)
        if not title:
            title = humanize_identifier(example.stem)
        if not summary:
            summary = default_example_summary(example)
        group, subgroup = infer_example_group(example)
        entries.append(
            ExampleEntry(
                title=title,
                summary=summary,
                group=group,
                subgroup=subgroup,
                path=example,
            )
        )
    return entries


def skill_entries() -> list[SkillEntry]:
    entries: list[SkillEntry] = []
    for skill_md in sorted(SKILLS.glob("*/SKILL.md")):
        metadata = parse_frontmatter(skill_md)
        resolved = {key: skill_frontmatter_value(metadata, key) for key in ("name", "description", *SKILL_TAXONOMY_KEYS)}
        missing = [key for key, value in resolved.items() if not value]
        if missing:
            raise SystemExit(
                f"{relative(skill_md)} is missing skill frontmatter fields: {', '.join(missing)}"
            )
        if resolved["kind"] not in SKILL_KIND_LABELS:
            raise SystemExit(f"{relative(skill_md)} has unsupported skill kind: {resolved['kind']}")
        if resolved["surface"] not in SKILL_SURFACE_LABELS:
            raise SystemExit(
                f"{relative(skill_md)} has unsupported skill surface: {resolved['surface']}"
            )
        if resolved["stage"] not in SKILL_STAGE_LABELS:
            raise SystemExit(
                f"{relative(skill_md)} has unsupported skill stage: {resolved['stage']}"
            )
        if resolved["trust"] not in SKILL_TRUST_LABELS:
            raise SystemExit(
                f"{relative(skill_md)} has unsupported skill trust: {resolved['trust']}"
            )
        entries.append(
            SkillEntry(
                name=resolved["name"],
                description=resolved["description"],
                kind=resolved["kind"],
                surface=resolved["surface"],
                stage=resolved["stage"],
                trust=resolved["trust"],
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
    entries = example_entries()
    lines = [
        "---",
        "title: Example Index",
        "summary: Generated catalog of runnable examples grouped by backend and workflow.",
        "---",
        "",
        "# Example Index",
        "",
        "This generated index groups runnable examples by backend and workflow and",
        "extracts a short summary from each example when available.",
        "",
    ]
    for group, subgroups in EXAMPLE_GROUPS:
        lines.extend([f"## {group}", ""])
        grouped_entries = [entry for entry in entries if entry.group == group]
        for subgroup in subgroups:
            subgroup_entries = [entry for entry in grouped_entries if entry.subgroup == subgroup]
            if not subgroup_entries:
                continue
            lines.extend([f"### {subgroup}", ""])
            for entry in subgroup_entries:
                lines.append(
                    f"- **{entry.title}**: {entry.summary}. "
                    f"`python {relative(entry.path)}` (`{relative(entry.path)}`)"
                )
            lines.append("")
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
