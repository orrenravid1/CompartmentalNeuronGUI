from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "compneurovis"
DOCS_REF = ROOT / "docs" / "reference"
EXAMPLES = ROOT / "examples"
SKILLS = ROOT / "skills"


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


def skill_entries() -> list[tuple[str, str, Path]]:
    entries: list[tuple[str, str, Path]] = []
    for skill_md in sorted(SKILLS.glob("*/SKILL.md")):
        name = ""
        description = ""
        lines = skill_md.read_text(encoding="utf-8").splitlines()
        in_frontmatter = False
        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                in_frontmatter = not in_frontmatter
                continue
            if not in_frontmatter:
                continue
            if stripped.startswith("name:"):
                name = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("description:"):
                description = stripped.split(":", 1)[1].strip()
        entries.append((name, description, skill_md))
    return entries


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
    for name, _, skill_md in skill_entries():
        lines.append(f"- `{name}`: `{relative(skill_md)}`")
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
    lines = [
        "---",
        "title: Skill Index",
        "summary: Generated list of repo-owned skills and their trigger descriptions.",
        "---",
        "",
        "# Skill Index",
        "",
    ]
    for name, description, skill_md in skill_entries():
        lines.append(f"- `{name}`: {description} (`{relative(skill_md)}`)")
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
