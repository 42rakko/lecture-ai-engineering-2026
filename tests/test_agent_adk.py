"""ADK 版学習支援エージェントの配線テスト。

実際の LLM API を呼ばない偽モデル（FakeLlm）を使い、
「ツールが登録され、呼ばれ、結果がモデルに戻り、最終回答が得られるか」
「MCP サーバー経由で接頭辞付きツールが呼ばれ、正しく動くか」
「使用量が正しく集計されるか」を確かめる。
"""

import argparse
import asyncio
import json
import subprocess
import time
from typing import AsyncGenerator

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams, StreamableHTTPConnectionParams
from google.genai import types
from mcp import StdioServerParameters

from app import agent_adk, tools
from app.agent_adk import (
    build_agent,
    build_mcp_agent,
    load_mcp_servers,
    print_tools,
    run_agent,
    summarize_usage,
)

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _usage(prompt: int, candidates: int) -> types.GenerateContentResponseUsageMetadata:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=prompt + candidates,
    )


class FakeLlm(BaseLlm):
    """指定したツールを1回呼び、その結果を受け取ったら最終回答を返す偽 LLM（API は呼ばない）。"""

    tool_name: str = "search_glossary"

    def __init__(self, tool_name: str = "search_glossary"):
        super().__init__(model="fake-model", tool_name=tool_name)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        got_result = any(
            part.function_response for content in llm_request.contents for part in content.parts or []
        )
        if not got_result:
            call = types.FunctionCall(name=self.tool_name, args={"query": "LoRA"})
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(function_call=call)]),
                usage_metadata=_usage(12, 6),
            )
        else:
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="LoRAについての解説です。第2回")]),
                usage_metadata=_usage(25, 15),
            )


def _child_pids(pattern: str) -> set[int]:
    """コマンドラインが pattern に合うプロセスの PID を返す（macOS / Linux 共通の pgrep を使う）。"""
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
    return {int(pid) for pid in out.split()}


async def _wait_until_no_new(before: set[int], pattern: str, timeout: float = 5.0) -> set[int]:
    """サブプロセスが止まるのを少し待ち、実行前になかった PID が残っていれば返す。"""
    deadline = time.monotonic() + timeout
    while True:
        leftover = _child_pids(pattern) - before
        if not leftover or time.monotonic() > deadline:
            return leftover
        await asyncio.sleep(0.2)


def test_base_tools_are_registered():
    names = {t.__name__ for t in build_agent().tools}
    assert BASE_TOOLS <= names  # 演習でツールを足しても壊れないよう「含む」で判定


def test_tool_call_result_and_answer_are_wired():
    answer, events = asyncio.run(run_agent(build_agent(model=FakeLlm()), "LoRAとは？"))

    assert "第2回" in answer
    calls = [c for e in events for c in e.get_function_calls()]
    returns = [r for e in events for r in e.get_function_responses()]
    assert [c.name for c in calls] == ["search_glossary"]
    assert [r.name for r in returns] == ["search_glossary"]

    usage = summarize_usage(events)
    assert (usage.requests, usage.input_tokens, usage.output_tokens, usage.tool_calls) == (2, 37, 21, 1)


def test_print_tools_lists_function_tools(capsys):
    asyncio.run(print_tools(build_agent()))
    out = capsys.readouterr().out
    assert "[search_glossary]" in out
    assert "[get_lecture_summary]" in out


def test_load_mcp_servers_builds_stdio_and_http(tmp_path):
    config = {
        "mcpServers": {
            "stdio_server": {"command": "python", "args": ["-m", "app.mcp_server"], "env": {"FOO": "BAR"}},
            "http_server": {"url": "http://localhost:8080/mcp"},
        }
    }
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    by_prefix = {t.tool_name_prefix: t for t in load_mcp_servers(path)}
    assert set(by_prefix) == {"stdio_server", "http_server"}

    stdio = by_prefix["stdio_server"].connection_params
    assert isinstance(stdio, StdioConnectionParams)
    assert stdio.server_params.command == "python"
    assert stdio.server_params.args == ["-m", "app.mcp_server"]
    assert stdio.server_params.env == {"FOO": "BAR"}

    http = by_prefix["http_server"].connection_params
    assert isinstance(http, StreamableHTTPConnectionParams)
    assert http.url == "http://localhost:8080/mcp"


def test_mcp_tool_call_reaches_the_model_via_real_server():
    """偽 LLM + 実際の MCP サーバー: 接頭辞付きツールの呼び出し → サーバーの結果 → 最終回答。"""

    async def run():
        toolsets = load_mcp_servers()
        try:
            agent = build_mcp_agent(model=FakeLlm("glossary_search_glossary"), toolsets=toolsets)
            return await run_agent(agent, "LoRAとは？")
        finally:
            for toolset in toolsets:
                await toolset.close()

    answer, events = asyncio.run(run())
    assert "第2回" in answer
    calls = [c for e in events for c in e.get_function_calls()]
    returns = [r for e in events for r in e.get_function_responses()]
    assert [c.name for c in calls] == ["glossary_search_glossary"]
    assert [r.name for r in returns] == ["glossary_search_glossary"]
    # サーバーが tools.search_glossary と同じ結果を返している（質問の語が含まれるだけでは通らない照合）
    definition = next(e for e in tools.load_glossary() if e["term"] == "LoRA")["definition"]
    assert definition in str(returns[0].response)


def _args(**overrides) -> argparse.Namespace:
    values = {"question": "LoRAとは？", "trace": False, "mcp": False, "list_tools": False}
    return argparse.Namespace(**{**values, **overrides})


def test_cli_prints_answer_before_trace(monkeypatch, capsys):
    monkeypatch.setattr(agent_adk, "build_model", lambda: FakeLlm())
    asyncio.run(agent_adk.amain(_args(trace=True)))
    out = capsys.readouterr().out
    # agent.py と同じく、最終回答 → trace → usage の順
    assert out.index("第2回") < out.index("--- trace ---") < out.index("[tool call]") < out.index("[usage]")
    assert "[usage] RunUsage(input_tokens=37, output_tokens=21, requests=2, tool_calls=1)" in out


def test_cli_mcp_runs_with_resolved_tools_and_stops_server(monkeypatch, capsys):
    """CLI の MCP 経路（先にツールを取得して渡す）が動き、終了後にサーバーのプロセスが残らない。"""
    monkeypatch.setattr(agent_adk, "build_model", lambda: FakeLlm("glossary_search_glossary"))
    before = _child_pids("app.mcp_server")

    async def run():
        await agent_adk.amain(_args(mcp=True, trace=True))
        # イベントループが生きているうちに確認する（ループ終了時の自動クリーンアップに頼らない）
        return await _wait_until_no_new(before, "app.mcp_server")

    leftover = asyncio.run(run())

    out = capsys.readouterr().out
    assert "第2回" in out
    assert "[tool call]   glossary_search_glossary(" in out
    assert leftover == set()


def test_cli_mcp_list_tools_shows_prefixed_names(capsys):
    asyncio.run(agent_adk.amain(_args(mcp=True, list_tools=True)))
    out = capsys.readouterr().out
    assert "[glossary_search_glossary]" in out


def test_cli_mcp_connection_failure_exits_with_message(monkeypatch):
    broken = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command="no-such-command-for-test"), timeout=3.0
        ),
        tool_name_prefix="broken",
    )
    monkeypatch.setattr(agent_adk, "load_mcp_servers", lambda: [broken])
    with pytest.raises(SystemExit) as exc:
        asyncio.run(agent_adk.amain(_args(mcp=True, list_tools=True)))
    assert "broken" in str(exc.value)


class _Event:
    def __init__(self, metadata=None, partial=False, tool_calls=0):
        self.usage_metadata = metadata
        self.partial = partial
        self._tool_calls = tool_calls

    def get_function_calls(self):
        return [object()] * self._tool_calls


def test_summarize_usage_skips_partial_and_missing_metadata():
    events = [
        _Event(_usage(100, 50), tool_calls=2),
        _Event(_usage(20, 10), partial=True, tool_calls=5),  # ストリーミングの途中イベントは数えない
        _Event(_usage(200, 80)),
        _Event(None),  # usage_metadata なし
    ]
    usage = summarize_usage(events)
    assert (usage.requests, usage.input_tokens, usage.output_tokens, usage.tool_calls) == (2, 300, 130, 2)
    # Pydantic AI の RunUsage と同じ見た目で表示される
    assert str(usage) == "RunUsage(input_tokens=300, output_tokens=130, requests=2, tool_calls=2)"


def test_summarize_usage_counts_thought_tokens_as_output():
    metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7, total_token_count=22
    )
    assert summarize_usage([_Event(metadata)]).output_tokens == 12
