from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "generate_mcp_configs.py"
SPEC = importlib.util.spec_from_file_location("generate_mcp_configs_script", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_mcp_configs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_mcp_configs)


def test_generate_mcp_configs_accepts_repo_source():
    payload = generate_mcp_configs._read_source()

    assert "mcpServers" in payload
    assert payload["mcpServers"]["fetch"]["command"] == "npx"


def test_generate_mcp_configs_rejects_http_url_transport_in_canonical_source():
    with pytest.raises(generate_mcp_configs.McpConfigError, match="unsupported key 'httpUrl'"):
        generate_mcp_configs.validate_servers(
            {
                "streaming": {
                    "httpUrl": "https://example.com/mcp",
                }
            }
        )


def test_generate_mcp_configs_rejects_unknown_server_keys():
    with pytest.raises(generate_mcp_configs.McpConfigError, match="unsupported key 'headers'"):
        generate_mcp_configs.validate_servers(
            {
                "remote": {
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer token"},
                }
            }
        )


def test_generate_mcp_configs_rejects_args_without_command():
    with pytest.raises(generate_mcp_configs.McpConfigError, match="field 'args' requires 'command'"):
        generate_mcp_configs.validate_servers(
            {
                "broken": {
                    "args": ["server"],
                }
            }
        )


def test_generate_mcp_configs_wraps_npx_for_portable_windows_stdio_launch():
    portable = generate_mcp_configs._portable_servers(
        {
            "fetch": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
            }
        }
    )

    fetch = portable["fetch"]
    assert fetch["command"] == "python"
    assert fetch["args"][:2] == ["-c", generate_mcp_configs.PORTABLE_STDIO_LAUNCHER]
    assert fetch["args"][2:] == ["npx", "-y", "@modelcontextprotocol/server-fetch"]


def test_generate_mcp_configs_leaves_non_npx_stdio_servers_unchanged():
    portable = generate_mcp_configs._portable_servers(
        {
            "arxiv": {
                "command": "uvx",
                "args": ["arxiv-mcp-server"],
            }
        }
    )

    assert portable["arxiv"]["command"] == "uvx"
    assert portable["arxiv"]["args"] == ["arxiv-mcp-server"]


def test_generate_mcp_configs_skips_unchanged_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "settings.json"
    target.write_text('{"same": true}\n', encoding="utf-8")

    def fail_write(self: Path, content: str, encoding: str = "utf-8") -> int:
        raise AssertionError("write_text should not be called for unchanged content")

    monkeypatch.setattr(Path, "write_text", fail_write)

    generate_mcp_configs._write(target, '{"same": true}\n')


def test_generate_mcp_configs_reports_all_locked_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(generate_mcp_configs, "ROOT", tmp_path)
    monkeypatch.setattr(
        generate_mcp_configs,
        "_TARGETS",
        [
            ("one.json", lambda servers: "one\n", lambda servers, path: False),
            ("two.json", lambda servers: "two\n", lambda servers, path: False),
        ],
    )
    monkeypatch.setattr(generate_mcp_configs, "_read_source", lambda: {"mcpServers": {}})

    def locked_write(path: Path, content: str) -> None:
        raise PermissionError(f"{path} is locked")

    monkeypatch.setattr(generate_mcp_configs, "_write", locked_write)

    with pytest.raises(generate_mcp_configs.McpConfigError) as exc_info:
        generate_mcp_configs.generate()

    message = str(exc_info.value)
    assert "one.json" in message
    assert "two.json" in message
