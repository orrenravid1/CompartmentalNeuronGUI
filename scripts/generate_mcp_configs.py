from __future__ import annotations

"""Generate agent-specific MCP config files from the canonical mcp.json source.

mcp.json at the repo root is the single source of truth. Each agent tool
expects its config in a different path and possibly a different format. This
script converts mcp.json into all of them so that adding, removing, or changing
a server only requires editing mcp.json and rerunning.

Supported agents:
  Claude Code       .claude/settings.json   (mcpServers key merged)
  VS Code/Copilot   .vscode/mcp.json        (standalone mcpServers JSON)
  Cursor            .cursor/mcp.json        (standalone mcpServers JSON)
  OpenAI Codex      .codex/config.toml      (TOML, stdio+HTTP)
  Gemini CLI        .gemini/settings.json   (mcpServers key merged)
  OpenCode          opencode.json           (mcp key, different structure)

Not generated (configured elsewhere):
  GitHub Copilot Cloud Agent — configured via GitHub repository Settings UI,
  not a file in the repo. See: Repository Settings > Copilot > Cloud agent.

Usage:
    python scripts/generate_mcp_configs.py          # write all agent configs
    python scripts/generate_mcp_configs.py --check  # fail if any are stale
"""

import argparse
import json
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mcp.json"
SUPPORTED_SERVER_KEYS = frozenset({"command", "args", "url", "env"})
PORTABLE_STDIO_LAUNCHER = (
    "import os,subprocess,sys; "
    "cmd=sys.argv[1:]; "
    "cmd=(['cmd','/c',*cmd] if os.name=='nt' and cmd and cmd[0].lower()=='npx' else cmd); "
    "raise SystemExit(subprocess.call(cmd))"
)


class McpConfigError(RuntimeError):
    """Raised when the canonical MCP configuration is invalid or unsupported."""


def validate_servers(servers: object) -> dict[str, dict]:
    if not isinstance(servers, dict):
        raise McpConfigError("mcpServers must be a JSON object")

    validated: dict[str, dict] = {}
    errors: list[str] = []
    for name, raw_cfg in servers.items():
        if not isinstance(name, str) or not name:
            errors.append(f"invalid server name {name!r}: names must be non-empty strings")
            continue
        if not isinstance(raw_cfg, dict):
            errors.append(f"server '{name}' must be a JSON object")
            continue

        unknown_keys = sorted(set(raw_cfg) - SUPPORTED_SERVER_KEYS)
        for key in unknown_keys:
            if key == "httpUrl":
                errors.append(
                    f"server '{name}' uses unsupported key 'httpUrl'; "
                    "canonical mcp.json currently supports only the portable subset "
                    "{command, args, url, env}"
                )
            else:
                errors.append(
                    f"server '{name}' uses unsupported key '{key}'; "
                    "canonical mcp.json currently supports only {command, args, url, env}"
                )

        transport_keys = [key for key in ("command", "url") if key in raw_cfg]
        if len(transport_keys) != 1:
            errors.append(
                f"server '{name}' must declare exactly one transport key: 'command' or 'url'"
            )

        command = raw_cfg.get("command")
        if command is not None and not isinstance(command, str):
            errors.append(f"server '{name}' field 'command' must be a string")

        if "args" in raw_cfg:
            args = raw_cfg["args"]
            if "command" not in raw_cfg:
                errors.append(f"server '{name}' field 'args' requires 'command'")
            elif not isinstance(args, list) or not all(isinstance(part, str) for part in args):
                errors.append(f"server '{name}' field 'args' must be a list of strings")

        url = raw_cfg.get("url")
        if url is not None and not isinstance(url, str):
            errors.append(f"server '{name}' field 'url' must be a string")

        if "env" in raw_cfg:
            env = raw_cfg["env"]
            if not isinstance(env, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in env.items()
            ):
                errors.append(f"server '{name}' field 'env' must be an object of string:string pairs")

        validated[name] = dict(raw_cfg)

    if errors:
        raise McpConfigError("invalid mcp.json:\n" + "\n".join(f"- {error}" for error in errors))
    return validated


def _read_source() -> dict:
    try:
        payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"invalid JSON in {SOURCE.name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise McpConfigError(f"{SOURCE.name} must contain a top-level JSON object")
    if "mcpServers" not in payload:
        raise McpConfigError(f"{SOURCE.name} must contain a top-level 'mcpServers' object")

    return {"mcpServers": validate_servers(payload["mcpServers"])}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                return
        except OSError:
            pass
    path.write_text(content, encoding="utf-8")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _merge_json(path: Path, key: str, value: object) -> str:
    """Read existing JSON file (if any), set key to value, return serialized result."""
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data[key] = value
    return json.dumps(data, indent=2) + "\n"


# ── renderers ──────────────────────────────────────────────────────────────────


def _render_json_mcp(servers: dict) -> str:
    """Standalone mcpServers JSON file (VS Code/Copilot, Cursor)."""
    return json.dumps({"mcpServers": _portable_servers(servers)}, indent=2) + "\n"


def _render_claude(servers: dict) -> str:
    """Merge mcpServers into .claude/settings.json, preserving other keys."""
    return _merge_json(ROOT / ".claude" / "settings.json", "mcpServers", _portable_servers(servers))


def _render_gemini(servers: dict) -> str:
    """Merge mcpServers into .gemini/settings.json, preserving other keys.

    Gemini uses the same mcpServers JSON schema as VS Code and Cursor.
    The canonical mcp.json intentionally stays on the portable subset
    {command, args, url, env}. If a future transport needs client-specific
    keys such as Gemini's httpUrl, extend every renderer first instead of
    adding that key to the canonical source.
    See: https://geminicli.com/docs/tools/mcp-server/
    """
    return _merge_json(ROOT / ".gemini" / "settings.json", "mcpServers", _portable_servers(servers))


def _opencode_mcp_value(servers: dict) -> dict:
    """Convert canonical servers dict to OpenCode's mcp section format.

    Differences from the canonical format:
    - command + args are merged into a single "command" array
    - env vars use "environment" instead of "env"
    - each server declares an explicit "type": "local" (stdio) or "remote" (HTTP/SSE)
    See: https://opencode.ai/docs/mcp-servers/
    """
    mcp: dict = {}
    for name, cfg in _portable_servers(servers).items():
        entry: dict = {"enabled": True}
        if "command" in cfg:
            entry["type"] = "local"
            entry["command"] = [cfg["command"]] + list(cfg.get("args", []))
        elif "url" in cfg:
            entry["type"] = "remote"
            entry["url"] = cfg["url"]
        if "env" in cfg:
            entry["environment"] = cfg["env"]
        mcp[name] = entry
    return mcp


def _render_opencode(servers: dict) -> str:
    """Merge mcp section into opencode.json, preserving other keys."""
    return _merge_json(ROOT / "opencode.json", "mcp", _opencode_mcp_value(servers))


def _render_codex_toml(servers: dict) -> str:
    """Convert to .codex/config.toml for OpenAI Codex (stdio and HTTP servers).

    Stdio servers use command/args; HTTP/SSE servers use url. Both are
    supported in the same file.
    See: https://developers.openai.com/codex/mcp
    """
    lines: list[str] = [
        "# Generated from mcp.json — edit mcp.json, then run:",
        "#   python scripts/generate_mcp_configs.py",
        "",
    ]
    for name, cfg in _portable_servers(servers).items():
        lines.append(f"[mcp_servers.{name}]")
        if "command" in cfg:
            lines.append(f"command = {json.dumps(cfg['command'])}")
        if "args" in cfg:
            toml_args = "[" + ", ".join(json.dumps(a) for a in cfg["args"]) + "]"
            lines.append(f"args = {toml_args}")
        if "url" in cfg:
            lines.append(f"url = {json.dumps(cfg['url'])}")
        if "env" in cfg:
            for k, v in cfg["env"].items():
                lines.append(f"{k} = {json.dumps(v)}")
        lines.append("")
    return "\n".join(lines)


def _portable_stdio_server(cfg: dict) -> dict:
    portable = dict(cfg)
    if portable.get("command") != "npx":
        return portable
    portable["args"] = ["-c", PORTABLE_STDIO_LAUNCHER, "npx", *portable.get("args", [])]
    portable["command"] = "python"
    return portable


def _portable_servers(servers: dict) -> dict:
    return {
        name: _portable_stdio_server(cfg)
        for name, cfg in servers.items()
    }


# ── check helpers ──────────────────────────────────────────────────────────────


def _full_check(renderer: Callable[[dict], str]) -> Callable[[dict, Path], bool]:
    """File is up to date when its entire content matches the renderer output."""
    return lambda servers, path: (
        path.exists() and path.read_text(encoding="utf-8") == renderer(servers)
    )


def _key_check(
    key: str, value_fn: Callable[[dict], object] | None = None
) -> Callable[[dict, Path], bool]:
    """File is up to date when one JSON key matches the expected value.

    Used for settings files we partially own (Claude, Gemini, OpenCode), where
    other keys in the file belong to the user or other tools.
    """
    def check(servers: dict, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            expected = value_fn(servers) if value_fn else servers
            return current.get(key) == expected
        except (json.JSONDecodeError, OSError):
            return False
    return check


# ── targets ────────────────────────────────────────────────────────────────────

# (relative path, renderer, check_fn)
_TARGETS: list[tuple[str, Callable[[dict], str], Callable[[dict, Path], bool]]] = [
    (".claude/settings.json",  _render_claude,     _key_check("mcpServers", _portable_servers)),
    (".vscode/mcp.json",       _render_json_mcp,   _full_check(_render_json_mcp)),
    (".cursor/mcp.json",       _render_json_mcp,   _full_check(_render_json_mcp)),
    (".codex/config.toml",     _render_codex_toml, _full_check(_render_codex_toml)),
    (".gemini/settings.json",  _render_gemini,     _key_check("mcpServers", _portable_servers)),
    ("opencode.json",          _render_opencode,   _key_check("mcp", _opencode_mcp_value)),
]


# ── generate / check ───────────────────────────────────────────────────────────


def generate() -> None:
    servers = _read_source()["mcpServers"]
    rendered = [(ROOT / rel, renderer(servers)) for rel, renderer, _ in _TARGETS]
    write_failures: list[str] = []
    for path, content in rendered:
        try:
            _write(path, content)
        except PermissionError:
            write_failures.append(_relative(path))
            continue
        except OSError as exc:
            raise McpConfigError(f"unable to write {_relative(path)}: {exc}") from exc
        print(f"  wrote {_relative(path)}")
    if write_failures:
        raise McpConfigError(
            "unable to write generated MCP config(s); another tool may be holding them open:\n"
            + "\n".join(f"- {path}" for path in write_failures)
        )


def check() -> None:
    servers = _read_source()["mcpServers"]
    stale = [rel for rel, _, check_fn in _TARGETS if not check_fn(servers, ROOT / rel)]
    if stale:
        raise SystemExit(
            "MCP configs are stale (run python scripts/generate_mcp_configs.py):\n"
            + "".join(f"  {p}\n" for p in stale)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any generated config is out of date.",
    )
    args = parser.parse_args()
    try:
        if args.check:
            check()
        else:
            generate()
    except McpConfigError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
