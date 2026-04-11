from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "pr_readiness.py"
SPEC = importlib.util.spec_from_file_location("pr_readiness_script", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pr_readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pr_readiness)

NOOP_COMMANDS = (("python", "-c", "print('ok')"),)


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "CompNeuroVis Tests")
    git(repo, "config", "user.email", "tests@example.com")
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("candidate\n", encoding="utf-8")
    git(repo, "add", ".gitignore", "tracked.txt")
    git(repo, "commit", "-m", "feat: candidate change")
    return repo


def write_and_commit_attestation(repo: Path) -> dict[str, object]:
    payload = pr_readiness.build_attestation(repo, commands=NOOP_COMMANDS)
    pr_readiness.write_attestation(repo, payload)
    trailers = payload["expected_final_commit"]["trailers"]
    git(repo, "add", payload["attestation_path"])
    git(
        repo,
        "commit",
        "-m",
        payload["expected_final_commit"]["subject"],
        "-m",
        f"PR-Readiness-For: {trailers['PR-Readiness-For']}",
        "-m",
        f"PR-Readiness-Tree: {trailers['PR-Readiness-Tree']}",
    )
    return payload


def test_verify_attestation_accepts_attestation_only_final_commit(tmp_path: Path):
    repo = init_repo(tmp_path)
    payload = write_and_commit_attestation(repo)

    verified = pr_readiness.verify_attestation(repo, required_commands=NOOP_COMMANDS)

    assert verified["candidate_commit"] == payload["candidate_commit"]


def test_verify_attestation_rejects_final_commit_with_extra_changes(tmp_path: Path):
    repo = init_repo(tmp_path)
    payload = pr_readiness.build_attestation(repo, commands=NOOP_COMMANDS)
    pr_readiness.write_attestation(repo, payload)
    (repo / "tracked.txt").write_text("candidate\nextra\n", encoding="utf-8")
    trailers = payload["expected_final_commit"]["trailers"]
    git(repo, "add", payload["attestation_path"], "tracked.txt")
    git(
        repo,
        "commit",
        "-m",
        payload["expected_final_commit"]["subject"],
        "-m",
        f"PR-Readiness-For: {trailers['PR-Readiness-For']}",
        "-m",
        f"PR-Readiness-Tree: {trailers['PR-Readiness-Tree']}",
    )

    with pytest.raises(pr_readiness.ReadinessError, match="exactly one attestation file"):
        pr_readiness.verify_attestation(repo, required_commands=NOOP_COMMANDS)


def test_verify_attestation_rejects_mismatched_commit_trailers(tmp_path: Path):
    repo = init_repo(tmp_path)
    payload = pr_readiness.build_attestation(repo, commands=NOOP_COMMANDS)
    pr_readiness.write_attestation(repo, payload)
    git(repo, "add", payload["attestation_path"])
    git(
        repo,
        "commit",
        "-m",
        payload["expected_final_commit"]["subject"],
        "-m",
        "PR-Readiness-For: deadbeef",
        "-m",
        "PR-Readiness-Tree: deadbeef",
    )

    with pytest.raises(pr_readiness.ReadinessError, match="trailer mismatch"):
        pr_readiness.verify_attestation(repo, required_commands=NOOP_COMMANDS)


def test_seal_with_commit_creates_valid_final_commit(tmp_path: Path):
    repo = init_repo(tmp_path)

    payload = pr_readiness.seal(repo, commands=NOOP_COMMANDS, commit_attestation=True)
    verified = pr_readiness.verify_attestation(repo, required_commands=NOOP_COMMANDS)

    assert verified["candidate_commit"] == payload["candidate_commit"]
    assert pr_readiness.commit_subject(repo) == payload["expected_final_commit"]["subject"]
    assert pr_readiness.working_tree_is_clean(repo)
