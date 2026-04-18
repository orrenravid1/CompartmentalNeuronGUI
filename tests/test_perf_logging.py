import json

from compneurovis import DiagnosticsSpec
from compneurovis._perf import clear_perf_logging_configuration, configure_perf_logging, perf_log


def _single_record(log_dir):
    paths = list(log_dir.glob("*.jsonl"))
    assert len(paths) == 1
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_diagnostics_spec_enables_perf_logging_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("COMPNV_PERF_LOG", raising=False)
    monkeypatch.delenv("COMPNV_PERF_STDERR", raising=False)
    clear_perf_logging_configuration()
    try:
        configure_perf_logging(DiagnosticsSpec(perf_log_enabled=True, perf_log_dir=tmp_path))
        perf_log("frontend", "demo", value=3)
        record = _single_record(tmp_path)
        assert record["component"] == "frontend"
        assert record["event"] == "demo"
        assert record["value"] == 3
    finally:
        clear_perf_logging_configuration()


def test_env_fallback_still_works_without_app_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPNV_PERF_LOG", str(tmp_path))
    monkeypatch.delenv("COMPNV_PERF_STDERR", raising=False)
    clear_perf_logging_configuration()
    try:
        perf_log("transport", "demo")
        record = _single_record(tmp_path)
        assert record["component"] == "transport"
        assert record["event"] == "demo"
    finally:
        clear_perf_logging_configuration()


def test_explicit_diagnostics_can_disable_env_fallback(monkeypatch, tmp_path):
    env_log_dir = tmp_path / "env-fallback"
    monkeypatch.setenv("COMPNV_PERF_LOG", str(env_log_dir))
    monkeypatch.delenv("COMPNV_PERF_STDERR", raising=False)
    clear_perf_logging_configuration()
    try:
        configure_perf_logging(DiagnosticsSpec())
        perf_log("transport", "should_not_log")
        assert not env_log_dir.exists()
    finally:
        clear_perf_logging_configuration()
