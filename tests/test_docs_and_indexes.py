from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_MARKDOWN_FILES = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "CHANGELOG.md",
)
CONTRIB_INSTALL_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "getting-started.md",
)
SKILL_REQUIRED_TOP_LEVEL_FRONTMATTER_KEYS = ("name", "description", "metadata")
SKILL_REQUIRED_TAXONOMY_METADATA_KEYS = ("kind", "surface", "stage", "trust")
SKILL_ALLOWED_TOP_LEVEL_FRONTMATTER_KEYS = {"name", "description", "license", "allowed-tools", "metadata"}
SKILL_KIND_VALUES = {"authoring", "coverage", "debug", "orchestration", "meta"}
SKILL_SURFACE_VALUES = {"backend", "cross-cutting", "docs", "examples", "frontend", "repo-infra"}
SKILL_STAGE_VALUES = {"debug", "explore", "implement", "release", "verify"}
SKILL_TRUST_VALUES = {"general", "maintainer-only", "proposal-only"}
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
MAX_SKILL_NAME_LENGTH = 64
AGENT_PRODUCT_NAMES = ("codex", "claude", "gemini", "perplexity")
FORBIDDEN_BARE_MKDOCS_COMMAND_PATTERNS = (
    re.compile(r"(?<!python -m )mkdocs serve\b"),
    re.compile(r"(?<!python -m )mkdocs build --strict\b"),
)
README_PUBLIC_API_SECTION_PATTERN = re.compile(r"^## Public API\n(?P<body>.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
AGENTS_STABLE_PACKAGE_MAP_PATTERN = re.compile(r"^## Stable Package Map\n(?P<body>.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
AGENTS_STABLE_PACKAGE_ENTRY_PATTERN = re.compile(r"^- `(?P<path>src/compneurovis/[^`]+)`:", re.MULTILINE)
README_PUBLIC_API_REQUIRED_NAMES = (
    "NeuronSession",
    "JaxleySession",
    "ReplaySession",
    "HistoryCaptureMode",
    "build_neuron_app",
    "build_jaxley_app",
    "build_surface_app",
    "build_replay_app",
    "grid_field",
    "run_app",
)
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+)\)")
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
ROOT_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.:-])"
    r"(?P<path>"
    r"AGENTS\.md|README\.md|CHANGELOG\.md|LICENSE|pyproject\.toml|poetry\.lock|"
    r"(?:\.compneurovis|\.github|docs|skills|scripts|src|tests|examples|res)(?:/[A-Za-z0-9._-]+)*/?"
    r")"
    r"(?![A-Za-z0-9_.-])"
)
ABSOLUTE_LOCAL_PATH_PATTERN = re.compile(r"^(?:/[A-Za-z]:/|[A-Za-z]:[/\\])")


def markdown_files() -> list[Path]:
    files = [path for path in ROOT_MARKDOWN_FILES if path.exists()]
    if (ROOT / "scratch" / "README.md").exists():
        files.append(ROOT / "scratch" / "README.md")
    if (ROOT / "src").exists():
        files.extend(sorted((ROOT / "src").rglob("README.md")))
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    files.extend(sorted((ROOT / "skills").rglob("SKILL.md")))
    if (ROOT / ".github").exists():
        files.extend(sorted((ROOT / ".github").rglob("*.md")))
    if (ROOT / ".compneurovis").exists():
        files.extend(sorted((ROOT / ".compneurovis").rglob("*.md")))
    return files


def parse_frontmatter(text: str) -> dict[str, object]:
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


def test_public_api_index_generation_is_stable_without_optional_neuron_dependency():
    with tempfile.TemporaryDirectory() as temp_dir:
        sitecustomize = Path(temp_dir) / "sitecustomize.py"
        sitecustomize.write_text(
            (
                "import importlib.abc\n"
                "import importlib\n"
                "import sys\n"
                "\n"
                "class _BlockNeuron(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname == 'neuron' or fullname.startswith('neuron.'):\n"
                "            raise ModuleNotFoundError(\"No module named 'neuron'\")\n"
                "        if fullname == 'jaxley' or fullname.startswith('jaxley.'):\n"
                "            raise ModuleNotFoundError(\"No module named 'jaxley'\")\n"
                "        return None\n"
                "\n"
                "sys.meta_path.insert(0, _BlockNeuron())\n"
                "\n"
                "_import_module = importlib.import_module\n"
                "\n"
                "def _guarded_import_module(name, package=None):\n"
                "    if name == 'compneurovis':\n"
                "        raise AssertionError('generate_indexes.py should not import compneurovis at runtime')\n"
                "    return _import_module(name, package)\n"
                "\n"
                "importlib.import_module = _guarded_import_module\n"
            ),
            encoding="utf-8",
        )
        env = os.environ | {"PYTHONPATH": temp_dir}
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import runpy, sys; "
                    "sys.argv=['generate_indexes.py','--check']; "
                    "runpy.run_path('scripts/generate_indexes.py', run_name='__main__')"
                ),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )


def test_public_api_module_exposes_optional_backend_names_without_eager_import():
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import compneurovis

        expected = {
            "NeuronSceneBuilder",
            "NeuronSession",
            "build_neuron_app",
            "JaxleySceneBuilder",
            "JaxleySession",
            "build_jaxley_app",
        }
        assert expected.issubset(set(compneurovis.__all__))
        assert expected.issubset(set(dir(compneurovis)))
    finally:
        sys.path.pop(0)


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


def test_skill_frontmatter_includes_required_taxonomy_metadata():
    skill_files = sorted((ROOT / "skills").rglob("SKILL.md"))
    assert skill_files

    names: list[str] = []
    for path in skill_files:
        metadata = parse_frontmatter(path.read_text(encoding="utf-8"))
        missing_top_level = [key for key in SKILL_REQUIRED_TOP_LEVEL_FRONTMATTER_KEYS if not metadata.get(key)]
        assert not missing_top_level, (
            f"{path.relative_to(ROOT).as_posix()} missing skill frontmatter keys: {', '.join(missing_top_level)}"
        )

        unexpected_keys = sorted(set(metadata) - SKILL_ALLOWED_TOP_LEVEL_FRONTMATTER_KEYS)
        assert not unexpected_keys, (
            f"{path.relative_to(ROOT).as_posix()} has unsupported top-level skill frontmatter keys: "
            + ", ".join(unexpected_keys)
        )

        taxonomy = metadata["metadata"]
        assert isinstance(taxonomy, dict), f"{path.relative_to(ROOT).as_posix()} metadata must be a mapping"
        missing_taxonomy = [key for key in SKILL_REQUIRED_TAXONOMY_METADATA_KEYS if not taxonomy.get(key)]
        assert not missing_taxonomy, (
            f"{path.relative_to(ROOT).as_posix()} missing skill taxonomy metadata: {', '.join(missing_taxonomy)}"
        )

        name = str(metadata["name"]).strip()
        description = str(metadata["description"]).strip()

        assert name == path.parent.name, (
            f"{path.relative_to(ROOT).as_posix()} should use its folder name as the skill name"
        )
        assert SKILL_NAME_PATTERN.match(name), f"{path.relative_to(ROOT).as_posix()} skill name must be hyphen-case"
        assert not (name.startswith("-") or name.endswith("-") or "--" in name)
        assert len(name) <= MAX_SKILL_NAME_LENGTH
        assert "<" not in description and ">" not in description
        assert len(description) <= 1024

        assert taxonomy["kind"] in SKILL_KIND_VALUES
        assert taxonomy["surface"] in SKILL_SURFACE_VALUES
        assert taxonomy["stage"] in SKILL_STAGE_VALUES
        assert taxonomy["trust"] in SKILL_TRUST_VALUES
        names.append(name)

    assert len(names) == len(set(names)), "skill names must be unique"


def test_skill_markdown_is_ascii_only():
    skill_files = sorted((ROOT / "skills").rglob("SKILL.md"))
    assert skill_files

    violations: list[str] = []
    for path in skill_files:
        text = path.read_text(encoding="utf-8")
        for index, char in enumerate(text):
            if ord(char) <= 127:
                continue
            line = text.count("\n", 0, index) + 1
            violations.append(
                f"{path.relative_to(ROOT).as_posix()}:{line}: non-ASCII character {ascii(char)}"
            )
            break

    assert not violations, "skills must stay ASCII-only for cross-agent validator compatibility:\n" + "\n".join(violations)


def test_skill_markdown_stays_agent_neutral():
    skill_files = sorted((ROOT / "skills").rglob("SKILL.md"))
    assert skill_files

    violations: list[str] = []
    for path in skill_files:
        text = path.read_text(encoding="utf-8").lower()
        for product in AGENT_PRODUCT_NAMES:
            if product in text:
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()}: avoid agent-specific product name '{product}' in repo-owned skill text"
                )

    assert not violations, "repo-owned skills should stay agent-neutral:\n" + "\n".join(violations)


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


def test_stable_package_map_entries_have_package_readmes():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    match = AGENTS_STABLE_PACKAGE_MAP_PATTERN.search(text)
    assert match is not None, "AGENTS.md is missing a Stable Package Map section"

    package_paths = [entry.group("path") for entry in AGENTS_STABLE_PACKAGE_ENTRY_PATTERN.finditer(match.group("body"))]
    assert package_paths, "AGENTS.md Stable Package Map should list at least one package path"

    missing: list[str] = []
    for relative_path in package_paths:
        package_dir = ROOT.joinpath(*relative_path.split("/"))
        assert package_dir.is_dir(), f"Stable Package Map entry does not exist: {relative_path}"
        if not (package_dir / "README.md").exists():
            missing.append(relative_path)

    assert not missing, "Stable Package Map packages missing README.md:\n" + "\n".join(missing)


def test_generated_skill_index_is_grouped_by_taxonomy():
    text = (ROOT / "docs" / "reference" / "skill-index.md").read_text(encoding="utf-8")
    for heading in (
        "## By Kind",
        "## By Surface",
        "## By Workflow Stage",
        "## By Trust",
        "## Full Catalog",
    ):
        assert heading in text, f"skill index should include {heading}"


def test_generated_example_index_is_grouped_by_backend_and_workflow():
    text = (ROOT / "docs" / "reference" / "example-index.md").read_text(encoding="utf-8")
    for heading in (
        "## Live Simulation Backends",
        "## Custom Sessions and Solvers",
        "## Field and Surface Workflows",
        "## Debug and Architecture Probes",
        "### NEURON",
        "### Jaxley",
        "### Custom",
    ):
        assert heading in text, f"example index should include {heading}"


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


def test_changelog_tracks_current_package_version():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"(?P<version>[^"]+)"', pyproject, re.MULTILINE)
    assert version_match is not None, "pyproject.toml is missing a package version"
    version = version_match.group("version")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## Unreleased" in changelog, "CHANGELOG.md should keep an Unreleased section"
    assert re.search(rf"^## {re.escape(version)}\b", changelog, re.MULTILINE), (
        "CHANGELOG.md should contain a section for the current package version "
        f"{version}"
    )
