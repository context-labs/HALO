from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import halo_cli.main as cli_main
from engine.engine_config import EngineConfig
from halo_cli.main import _make_config, _parse_headers, cli


def test_help_exposes_provider_and_model_flags_without_no_telemetry() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--base-url" in result.output
    assert "--api-key" in result.output
    assert "-H, --header" in result.output
    assert "--default-header" not in result.output
    assert "--temperature" in result.output
    assert "--max-output-tokens" in result.output
    assert "--synthesis-model" in result.output
    assert "--compaction-model" in result.output
    assert "--parallel-tool-calls / --no-parallel-tool-calls" in result.output
    assert "--telemetry" in result.output
    assert "--no-telemetry" not in result.output
    assert "--repo-path" in result.output


def test_make_config_threads_cli_options_into_engine_config(tmp_path: Path) -> None:
    cfg = _make_config(
        model="claude-opus-4-7",
        synthesis_model=None,
        compaction_model=None,
        max_depth=3,
        max_turns=12,
        max_parallel=5,
        temperature=0.2,
        max_output_tokens=1024,
        parallel_tool_calls=False,
        reasoning_effort="low",
        refusal_retries=2,
        base_url="https://api.anthropic.com/v1/",
        api_key="sk-ant-test",
        default_headers={"anthropic-beta": "tools-2025-01-01"},
        repo_path=tmp_path,
    )

    assert cfg.maximum_depth == 3
    assert cfg.maximum_parallel_subagents == 5
    assert cfg.root_agent.maximum_turns == 12
    assert cfg.subagent.maximum_turns == 12
    assert cfg.root_agent.refusal_retries == 2
    assert cfg.subagent.refusal_retries == 2
    assert cfg.model_provider.base_url == "https://api.anthropic.com/v1/"
    assert cfg.model_provider.api_key == "sk-ant-test"
    assert cfg.model_provider.default_headers == {"anthropic-beta": "tools-2025-01-01"}
    assert cfg.repo_path == tmp_path

    for model in (cfg.root_agent.model, cfg.subagent.model):
        assert model.name == "claude-opus-4-7"
        assert model.temperature == 0.2
        assert model.maximum_output_tokens == 1024
        assert model.parallel_tool_calls is False
        assert model.reasoning_effort == "low"

    for model in (cfg.synthesis_model, cfg.compaction_model):
        assert model.name == "claude-opus-4-7"
        assert model.temperature == 0.2
        assert model.maximum_output_tokens == 1024
        assert model.parallel_tool_calls is False
        assert model.reasoning_effort is None


def test_make_config_overrides_synthesis_and_compaction_models() -> None:
    """Synthesis/compaction never inherit the CLI --reasoning-effort: it
    targets the agents' model, and a cheap non-reasoning override must
    not receive an unsupported parameter. ``effective_reasoning_effort``
    resolves each model's own family default at call time instead."""
    cfg = _make_config(
        model="claude-opus-4-7",
        synthesis_model="gpt-4.1-nano",
        compaction_model="gpt-4.1-nano",
        max_depth=3,
        max_turns=12,
        max_parallel=5,
        temperature=None,
        max_output_tokens=None,
        parallel_tool_calls=True,
        reasoning_effort="low",
        refusal_retries=0,
        base_url=None,
        api_key=None,
        default_headers=None,
        repo_path=None,
    )

    assert cfg.repo_path is None
    assert cfg.root_agent.model.name == "claude-opus-4-7"
    assert cfg.root_agent.model.reasoning_effort == "low"
    assert cfg.subagent.model.name == "claude-opus-4-7"
    assert cfg.subagent.model.reasoning_effort == "low"
    assert cfg.synthesis_model.name == "gpt-4.1-nano"
    assert cfg.synthesis_model.reasoning_effort is None
    assert cfg.synthesis_model.effective_reasoning_effort() is None
    assert cfg.compaction_model.name == "gpt-4.1-nano"
    assert cfg.compaction_model.reasoning_effort is None


def test_cli_leaves_provider_fields_unset_for_openai_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("")
    captured: dict[str, Any] = {}

    async def capture_stream(
        trace_path: Path,
        prompt: str,
        cfg: EngineConfig,
        *,
        telemetry: bool = False,
    ) -> None:
        captured["trace_path"] = trace_path
        captured["prompt"] = prompt
        captured["cfg"] = cfg
        captured["telemetry"] = telemetry

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    monkeypatch.setattr(cli_main, "_stream", capture_stream)

    result = CliRunner().invoke(cli, [str(trace_path), "--prompt", "hi"])

    assert result.exit_code == 0
    assert captured["trace_path"] == trace_path
    assert captured["prompt"] == "hi"
    assert captured["cfg"].model_provider.api_key is None
    assert captured["cfg"].model_provider.base_url is None
    assert captured["telemetry"] is False


def test_cli_threads_repo_path_into_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    captured: dict[str, Any] = {}

    async def capture_stream(
        trace_path: Path,
        prompt: str,
        cfg: EngineConfig,
        *,
        telemetry: bool = False,
    ) -> None:
        captured["cfg"] = cfg

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setattr(cli_main, "_stream", capture_stream)

    result = CliRunner().invoke(
        cli, [str(trace_path), "--prompt", "hi", "--repo-path", str(repo_dir)]
    )

    assert result.exit_code == 0
    assert captured["cfg"].repo_path == repo_dir


def test_cli_rejects_nonexistent_repo_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    result = CliRunner().invoke(
        cli, [str(trace_path), "--prompt", "hi", "--repo-path", str(tmp_path / "nope")]
    )

    assert result.exit_code != 0


def test_cli_repo_path_requires_ripgrep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setattr(cli_main, "find_ripgrep", lambda: None)

    result = CliRunner().invoke(
        cli, [str(trace_path), "--prompt", "hi", "--repo-path", str(repo_dir)]
    )

    assert result.exit_code == 1
    assert "ripgrep" in result.output


def test_parse_headers() -> None:
    assert _parse_headers(
        [
            "HTTP-Referer: https://example.com",
            "X-Title: HALO",
        ]
    ) == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "HALO",
    }


def test_parse_headers_rejects_invalid_header() -> None:
    with pytest.raises(typer.BadParameter):
        _parse_headers(["X-Title=HALO"])
