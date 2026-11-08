"""第3回: Google ADK（Agent Development Kit）で作る学習支援エージェント。

app/agent.py（Pydantic AI 版）と同じ機能を、Google ADK で書き直したもの。

使い方:
    uv run python -m app.agent_adk "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
    uv run python -m app.agent_adk --trace "LoRAとQLoRAの違いは？"   # ツール呼び出しを表示
    uv run python -m app.agent_adk --mcp --trace "LoRAとは？"        # 第5回: ツールを MCP サーバーから受け取る
    uv run python -m app.agent_adk --mcp --list-tools               # 第5回: モデルに見えるツール一覧（APIキー不要）

エージェント = モデル + ループ + ツール。
ADK では LlmAgent がモデルとツールを持ち、Runner がループ（モデル呼び出し → ツール実行 → …）を回す。
tools.py の関数はそのままツールとして渡せる（型ヒントと docstring からスキーマを作る）。
"""

import argparse
import asyncio
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams, StreamableHTTPConnectionParams
from google.genai import Client, errors, types
from mcp import StdioServerParameters

from app import tools
from app.config import MODEL, api_error_message, get_api_key

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

APP_NAME = "aie2026"
USER_ID = "student"

# モデルは作成時に渡す（ADK の LlmAgent はモデルを持つ。テストでは API を呼ばない偽 LLM に差し替える）
FUNCTION_TOOLS = [tools.search_glossary, tools.get_lecture_summary, tools.make_quiz]

# 第5回: 使う MCP サーバーの設定ファイル。Claude Desktop や Cursor と同じ "mcpServers" 形式
MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"


def _make_agent(model, agent_tools: list) -> LlmAgent:
    return LlmAgent(name="learning_assistant", model=model or MODEL, instruction=INSTRUCTIONS, tools=agent_tools)


def build_agent(model=None) -> LlmAgent:
    """tools.py の関数をそのままツールとして持つエージェント。model を省くと MODEL の名前から ADK が解決する。"""
    return _make_agent(model, list(FUNCTION_TOOLS))


def load_mcp_servers(config_path: Path = MCP_CONFIG) -> list[McpToolset]:
    """設定ファイル（mcpServers 形式）を読み、サーバーごとに接続（McpToolset）を作る。

    - "command" があれば stdio：サーバーをサブプロセスとして起動する
    - "url" があれば Streamable HTTP：起動済みのサーバーに接続する
    ツール名には「サーバー名_」を付ける（例: glossary_search_glossary）。複数のサーバーに
    同じ名前のツールがあっても区別できるようにするため。
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]
    toolsets = []
    for name, spec in config.items():
        if "url" in spec:
            params = StreamableHTTPConnectionParams(url=spec["url"])
        else:
            params = StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=spec["command"], args=spec.get("args", []), env=spec.get("env")
                )
            )
        toolsets.append(McpToolset(connection_params=params, tool_name_prefix=name))
    return toolsets


def build_mcp_agent(model=None, config_path: Path = MCP_CONFIG, toolsets: list[McpToolset] | None = None) -> LlmAgent:
    """第5回: 関数を直接登録する代わりに、MCP サーバーが公開するツールを使うエージェント。INSTRUCTIONS は同じ。

    ADK では接続（McpToolset）を終了時に close() する必要があるため、呼び出し側が toolsets を作って渡せる。
    """
    return _make_agent(model, toolsets if toolsets is not None else load_mcp_servers(config_path))


async def check_mcp_servers(toolsets: list[McpToolset]) -> None:
    """MCP サーバーに実際につないでツールを取得し、失敗したら理由を表示して終了する。

    ADK は、エージェント実行時の接続エラーを握りつぶして「ツールなし」で動いてしまう。
    それを避けるため、実行前にここで確かめる。
    """
    for toolset in toolsets:
        try:
            await toolset.get_tools_with_prefix()
        except Exception as exc:
            raise SystemExit(f"MCP サーバー {toolset.tool_name_prefix!r} のツール取得に失敗しました: {exc}") from None


def build_model() -> Gemini:
    """APIキーは環境変数 GEMINI_API_KEY から読む（config.get_api_key）。呼ぶまで鍵は要らない。"""
    return Gemini(model=MODEL, client=Client(api_key=get_api_key()))


async def print_tools(target: LlmAgent) -> None:
    """モデルに見えているツールの一覧（名前・説明文・引数）を表示する。LLM は呼ばない。

    ADK がモデルへ渡す「関数宣言」を直接読む。MCP サーバーのツールも、関数ツールも、
    モデルから見れば同じ形をしている。
    """
    declarations = []
    for tool in await target.canonical_tools():
        if (declaration := tool._get_declaration()) is not None:
            declarations.append(declaration)
    print(f"モデルに見えているツール: {len(declarations)} 個")
    for decl in declarations:
        print(f"\n[{decl.name}]")
        print("  " + inspect.cleandoc(decl.description or "").replace("\n", "\n  "))
        schema = decl.parameters_json_schema or (decl.parameters.model_dump(exclude_none=True) if decl.parameters else {})
        print(f"  引数: {json.dumps(schema.get('properties', {}), ensure_ascii=False, default=str)}")


@dataclass
class RunUsage:
    """質問全体の使用量。Pydantic AI の RunUsage と同じ項目・同じ表示にしてある（比較しやすくするため）。

    ADK には集計済みの値がなく、モデル応答ごとのイベントに使用量が付くので、summarize_usage() で合計する。
    """

    input_tokens: int = 0
    output_tokens: int = 0  # 回答 + 思考のトークン
    requests: int = 0  # LLM を呼んだ回数
    tool_calls: int = 0  # ツールが呼ばれた回数


def summarize_usage(events: list) -> RunUsage:
    """イベント一覧から、完了した LLM 呼び出しの使用量とツール呼び出し回数を合計する。"""
    usage = RunUsage()
    for event in events:
        if getattr(event, "partial", False):
            continue
        usage.tool_calls += len(event.get_function_calls())
        metadata = getattr(event, "usage_metadata", None)
        if metadata:
            usage.requests += 1
            usage.input_tokens += metadata.prompt_token_count or 0
            usage.output_tokens += (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0)
    return usage


def print_trace(events: list) -> None:
    """どのツールが、どの引数で呼ばれ、何を返したかを表示する。"""
    print("\n--- trace ---")
    for event in events:
        for call in event.get_function_calls():
            print(f"[tool call]   {call.name}({call.args})")
        for response in event.get_function_responses():
            print(f"[tool return] {response.name} -> {str(response.response)[:120]}…")
    print(f"[usage] {summarize_usage(events)}")


async def run_agent(target: LlmAgent, question: str) -> tuple[str, list]:
    """Runner でエージェントを1回走らせ、(最終回答, イベント一覧) を返す。"""
    runner = InMemoryRunner(agent=target, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    message = types.Content(role="user", parts=[types.Part(text=question)])

    events = []
    answer = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        events.append(event)
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(part.text or "" for part in event.content.parts)
    return answer, events


async def amain(args: argparse.Namespace) -> None:
    toolsets = load_mcp_servers() if args.mcp else None
    try:
        if toolsets:
            await check_mcp_servers(toolsets)

        def make_target(model=None) -> LlmAgent:
            return build_mcp_agent(model, toolsets=toolsets) if args.mcp else build_agent(model)

        if args.list_tools:
            await print_tools(make_target())
            return

        print(f"[model] {MODEL}")
        if args.mcp:
            print("[tools] MCP サーバー（app/mcp_servers.json）")
        try:
            answer, events = await run_agent(make_target(build_model()), args.question)
        except errors.APIError as e:
            print(api_error_message(e.code, str(e.message)[:200]))
            raise SystemExit(1)
        print(answer)
        if args.trace:
            print_trace(events)
    finally:
        for toolset in toolsets or []:  # MCP サーバーのサブプロセスを止める
            await toolset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="学習支援エージェントに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    parser.add_argument("--mcp", action="store_true", help="第5回: ツールを MCP サーバー（app/mcp_servers.json）から受け取る")
    parser.add_argument("--list-tools", action="store_true", help="モデルに見えるツール一覧を表示して終了する（APIキー不要）")
    args = parser.parse_args()

    if not args.list_tools and not args.question:
        parser.error("質問を指定してください")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
