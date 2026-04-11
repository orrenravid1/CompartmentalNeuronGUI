from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
ATTESTATION_DIR_REL_PATH = Path(".compneurovis/pr-readiness")
SCHEMA_VERSION = 1
DEFAULT_VERIFICATION_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("python", "scripts/check_architecture_invariants.py"),
    ("pytest",),
    ("python", "-m", "compileall", "src", "examples", "tests"),
    ("python", "scripts/generate_indexes.py", "--check"),
)


class ReadinessError(RuntimeError):
    """Raised when the PR readiness workflow is incomplete or inconsistent."""


def git(
    root: Path,
    *args: str,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=capture_output,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"git {' '.join(args)} failed with exit code {completed.returncode}"
        raise ReadinessError(detail)
    return completed


def ensure_repo_root(root: Path) -> None:
    actual = git(root, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(actual).resolve() != root.resolve():
        raise ReadinessError(f"expected git root {root}, found {actual}")


def working_tree_is_clean(root: Path) -> bool:
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    return not status.strip()


def head_commit(root: Path, rev: str = "HEAD") -> str:
    return git(root, "rev-parse", rev).stdout.strip()


def commit_tree(root: Path, rev: str = "HEAD") -> str:
    return git(root, "log", "-1", "--format=%T", rev).stdout.strip()


def commit_subject(root: Path, rev: str = "HEAD") -> str:
    return git(root, "log", "-1", "--format=%s", rev).stdout.rstrip("\n")


def commit_body(root: Path, rev: str = "HEAD") -> str:
    return git(root, "log", "-1", "--format=%B", rev).stdout


def parent_commit(root: Path, rev: str = "HEAD") -> str:
    tokens = git(root, "rev-list", "--parents", "-n", "1", rev).stdout.strip().split()
    if len(tokens) != 2:
        raise ReadinessError(f"{rev} must have exactly one parent commit")
    return tokens[1]


def attestation_rel_path(candidate_commit: str) -> Path:
    return ATTESTATION_DIR_REL_PATH / f"{candidate_commit}.json"


def attestation_path(root: Path, relative_path: str | Path) -> Path:
    return root / Path(relative_path)


def normalize_command(command: Sequence[str]) -> list[str]:
    return [sys.executable if part == "python" else part for part in command]


def serialize_commands(commands: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(command) for command in commands]


def parse_trailers(message: str) -> dict[str, str]:
    trailers: dict[str, str] = {}
    for line in message.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        trailers[key] = value
    return trailers


def expected_subject(candidate_commit: str) -> str:
    return f"chore(pr-readiness): attest {candidate_commit[:12]}"


def expected_trailers(candidate_commit: str, candidate_tree: str) -> dict[str, str]:
    return {
        "PR-Readiness-For": candidate_commit,
        "PR-Readiness-Tree": candidate_tree,
    }


def run_verification_commands(root: Path, commands: Sequence[Sequence[str]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for command in commands:
        actual = normalize_command(command)
        started = time.perf_counter()
        completed = subprocess.run(actual, cwd=root)
        duration = round(time.perf_counter() - started, 3)
        if completed.returncode != 0:
            rendered = " ".join(command)
            raise ReadinessError(f"verification command failed: {rendered}")
        results.append(
            {
                "argv": list(command),
                "duration_seconds": duration,
            }
        )
    return results


def check_readiness(
    root: Path,
    commands: Sequence[Sequence[str]] = DEFAULT_VERIFICATION_COMMANDS,
) -> list[dict[str, object]]:
    ensure_repo_root(root)
    return run_verification_commands(root, commands)


def build_attestation(
    root: Path,
    commands: Sequence[Sequence[str]] = DEFAULT_VERIFICATION_COMMANDS,
) -> dict[str, object]:
    ensure_repo_root(root)
    if not working_tree_is_clean(root):
        raise ReadinessError(
            "seal requires a clean git working tree; commit the implementation first, then rerun seal"
        )

    candidate_commit = head_commit(root)
    candidate_tree = commit_tree(root)
    relative_path = attestation_rel_path(candidate_commit)
    results = run_verification_commands(root, commands)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/pr_readiness.py",
        "workflow": "pr-readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attestation_path": relative_path.as_posix(),
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "candidate_subject": commit_subject(root),
        "expected_final_commit": {
            "subject": expected_subject(candidate_commit),
            "trailers": expected_trailers(candidate_commit, candidate_tree),
            "changed_paths": [relative_path.as_posix()],
        },
        "verification_commands": results,
    }


def write_attestation(root: Path, payload: dict[str, object]) -> Path:
    relative_path = payload.get("attestation_path")
    if not isinstance(relative_path, str):
        raise ReadinessError("attestation payload is missing attestation_path")
    path = attestation_path(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_attestation(root: Path, relative_path: str | Path) -> dict[str, object]:
    path = attestation_path(root, relative_path)
    if not path.exists():
        raise ReadinessError(f"missing attestation: {Path(relative_path).as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def commands_from_payload(payload: dict[str, object]) -> list[list[str]]:
    command_entries = payload.get("verification_commands")
    if not isinstance(command_entries, list):
        raise ReadinessError("attestation is missing verification_commands")

    commands: list[list[str]] = []
    for entry in command_entries:
        if not isinstance(entry, dict):
            raise ReadinessError("verification_commands must contain objects")
        argv = entry.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) for part in argv):
            raise ReadinessError("each verification command must contain a string argv list")
        commands.append(argv)
    return commands


def verify_attestation(
    root: Path,
    required_commands: Sequence[Sequence[str]] = DEFAULT_VERIFICATION_COMMANDS,
    rerun_commands: bool = False,
) -> dict[str, object]:
    ensure_repo_root(root)

    actual_paths = [
        line.strip()
        for line in git(root, "diff", "--name-only", "--no-renames", "HEAD^", "HEAD").stdout.splitlines()
        if line.strip()
    ]
    if len(actual_paths) != 1:
        raise ReadinessError("final PR readiness commit must change exactly one attestation file")
    actual_path = actual_paths[0]
    attestation_prefix = f"{ATTESTATION_DIR_REL_PATH.as_posix()}/"
    if not actual_path.startswith(attestation_prefix) or not actual_path.endswith(".json"):
        raise ReadinessError(
            "final PR readiness commit must add exactly one json attestation under "
            f"{ATTESTATION_DIR_REL_PATH.as_posix()}/"
        )

    payload = load_attestation(root, actual_path)

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ReadinessError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    if payload.get("attestation_path") != actual_path:
        raise ReadinessError("attestation_path does not match the attestation file committed at HEAD")

    parent = parent_commit(root)
    candidate_commit = payload.get("candidate_commit")
    candidate_tree = payload.get("candidate_tree")
    if candidate_commit != parent:
        raise ReadinessError("attestation must target HEAD^ so the seal commit is the explicit final commit")
    if candidate_tree != commit_tree(root, "HEAD^"):
        raise ReadinessError("attested tree hash does not match the parent commit tree")

    expected_paths = [actual_path]
    if actual_paths != expected_paths:
        raise ReadinessError(
            "final PR readiness commit must change only "
            f"{actual_path}, found: {actual_paths}"
        )

    expected_commit = payload.get("expected_final_commit")
    if not isinstance(expected_commit, dict):
        raise ReadinessError("attestation is missing expected_final_commit")
    if expected_commit.get("changed_paths") != expected_paths:
        raise ReadinessError("attestation changed_paths does not match the committed attestation path")

    expected_subject_value = expected_commit.get("subject")
    if expected_subject_value != expected_subject(parent):
        raise ReadinessError("attestation subject does not match the canonical PR readiness subject")
    if commit_subject(root) != expected_subject_value:
        raise ReadinessError("final commit subject does not match the attested PR readiness subject")

    expected_trailer_values = expected_commit.get("trailers")
    if expected_trailer_values != expected_trailers(parent, commit_tree(root, "HEAD^")):
        raise ReadinessError("attestation trailers do not match the canonical PR readiness trailers")

    actual_trailers = parse_trailers(commit_body(root))
    for key, value in expected_trailer_values.items():
        if actual_trailers.get(key) != value:
            raise ReadinessError(f"final commit trailer mismatch for {key}")

    recorded_commands = commands_from_payload(payload)
    if recorded_commands != serialize_commands(required_commands):
        raise ReadinessError("attestation verification command set does not match the required PR readiness profile")

    if rerun_commands:
        run_verification_commands(root, recorded_commands)

    return payload


def create_attestation_commit(
    root: Path,
    payload: dict[str, object],
    required_commands: Sequence[Sequence[str]] = DEFAULT_VERIFICATION_COMMANDS,
) -> str:
    relative_path = payload.get("attestation_path")
    if not isinstance(relative_path, str):
        raise ReadinessError("attestation payload is missing attestation_path")

    expected_commit = payload.get("expected_final_commit")
    if not isinstance(expected_commit, dict):
        raise ReadinessError("attestation payload is missing expected_final_commit")

    subject = expected_commit.get("subject")
    trailers = expected_commit.get("trailers")
    if not isinstance(subject, str) or not isinstance(trailers, dict):
        raise ReadinessError("attestation payload has malformed final commit metadata")

    git(root, "add", relative_path)
    git(
        root,
        "commit",
        "-m",
        subject,
        "-m",
        f"PR-Readiness-For: {trailers['PR-Readiness-For']}",
        "-m",
        f"PR-Readiness-Tree: {trailers['PR-Readiness-Tree']}",
    )
    verify_attestation(root, required_commands=required_commands)
    return head_commit(root)


def check(root: Path, commands: Sequence[Sequence[str]] = DEFAULT_VERIFICATION_COMMANDS) -> list[dict[str, object]]:
    results = check_readiness(root, commands=commands)
    print("PR readiness checks passed:")
    for entry in results:
        argv = entry["argv"]
        duration = entry["duration_seconds"]
        print(f"- {' '.join(argv)} ({duration:.3f}s)")
    return results


def seal(
    root: Path,
    commands: Sequence[Sequence[str]] = DEFAULT_VERIFICATION_COMMANDS,
    commit_attestation: bool = False,
) -> dict[str, object]:
    payload = build_attestation(root, commands=commands)
    path = write_attestation(root, payload)
    commit = str(payload["candidate_commit"])
    tree = str(payload["candidate_tree"])
    subject = str(payload["expected_final_commit"]["subject"])
    relative_path = str(payload["attestation_path"])
    print(f"Wrote {path.relative_to(root).as_posix()} for {commit[:12]}")
    if commit_attestation:
        final_commit = create_attestation_commit(root, payload, required_commands=commands)
        print(f"Created final PR readiness commit {final_commit[:12]} for {commit[:12]}")
        return payload
    print("Commit it as the explicit final PR commit with:")
    print(f"git add {relative_path}")
    print(
        'git commit '
        f'-m "{subject}" '
        f'-m "PR-Readiness-For: {commit}" '
        f'-m "PR-Readiness-Tree: {tree}"'
    )
    return payload


def verify(root: Path, rerun_commands: bool) -> None:
    payload = verify_attestation(root, rerun_commands=rerun_commands)
    candidate_commit = str(payload["candidate_commit"])
    suffix = " and reran verification commands" if rerun_commands else ""
    print(f"Verified PR readiness attestation for {candidate_commit[:12]}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal and verify CompNeuroVis PR readiness as an explicit final git commit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check",
        help="Run the canonical PR readiness verification commands without creating an attestation.",
    )

    seal_parser = subparsers.add_parser(
        "seal",
        help="Run the readiness verification set and write the attestation file.",
    )
    seal_parser.add_argument(
        "--commit",
        action="store_true",
        help="Create the required final attestation commit automatically after sealing.",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify that HEAD is an attestation-only final commit for the sealed parent commit.",
    )
    verify_parser.add_argument(
        "--rerun-commands",
        action="store_true",
        help="Rerun the attested verification commands after structural verification succeeds.",
    )

    args = parser.parse_args()
    try:
        if args.command == "check":
            check(ROOT)
        elif args.command == "seal":
            seal(ROOT, commit_attestation=args.commit)
        else:
            verify(ROOT, rerun_commands=args.rerun_commands)
    except ReadinessError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
