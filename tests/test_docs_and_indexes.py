from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_docs_and_skills_have_front_matter():
    markdown_files = list((ROOT / "docs").rglob("*.md")) + list((ROOT / "skills").rglob("SKILL.md"))
    assert markdown_files
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"missing front matter start: {path}"
        assert "\n---\n" in text[4:], f"missing front matter end: {path}"


def test_agents_links_resolve():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    candidates = re.findall(r"`([A-Za-z0-9._/-]+)`", agents)
    unresolved = []
    for candidate in candidates:
        if candidate.startswith("http"):
            continue
        if candidate.startswith("python"):
            continue
        if "/" not in candidate:
            continue
        if not (ROOT / candidate).exists():
            unresolved.append(candidate)
    assert not unresolved, f"unresolved AGENTS paths: {unresolved}"
