# 計画書: agent.py を Google ADK で書き直す（app/agent_adk.py）

## 目的

`app/agent.py`（Pydantic AI 版の学習支援エージェント）と同じ機能を、Google ADK（Agent Development Kit）で
実装した `app/agent_adk.py` を用意する。フレームワーク間の違いを比較できるようにするのが狙い。

- 既存の `app/agent.py` と `tests/test_agent.py` は変更しない（並存させる）
- ツール本体（`app/tools.py`）、MCP サーバー（`app/mcp_server.py`）、設定（`app/config.py`）は共用する

## 現状

- `app/agent_adk.py` は `agent.py` を元に書き直した初稿が置かれている（未検証・未コミット）
- 書き直しの際、ADK の API は一時 venv（scratchpad）に `google-adk 2.14.0` と `mcp` を入れて確認した
- プロジェクトの依存（`pyproject.toml`）に `google-adk` は無い。`pydantic-ai-slim[google,mcp]==2.57.0` のみ
- 初稿はまだ一度も実行していない（`--list-tools` の動作確認が未実施）
- 初稿に直すべき点がある:
  - `[usage]` が最後の `usage_metadata` だけの表示になっている（下記の仕様＝質問全体の集計に合わせる）
  - `--trace` をイベントループ内でリアルタイムに表示しており、`agent.py`（回答 → trace の順）と逆になっている
  - docstring の使い方が `app.agent_adm` のまま（`app.agent_adk` に直す）
  - `# TODO(第3回 演習)` のコメントが残っている（ADK 版には写さない方針。削除する）

## Pydantic AI → ADK の対応

| 役割 | Pydantic AI（agent.py） | ADK（agent_adk.py） |
|---|---|---|
| エージェント | `Agent(instructions=...)` | `LlmAgent(name, model, instruction, tools)` |
| 関数ツール登録 | `agent.tool_plain(f)` | `tools=[f, ...]`（関数をそのまま渡す） |
| モデル | `GoogleModel` + `GoogleProvider` | `Gemini(model, client=genai.Client(api_key))` |
| 実行 | `agent.run_sync(q, model=...)` | `InMemoryRunner.run_async(...)`（非同期・イベントのストリーム） |
| MCP | `MCPToolset(Client(...)).prefixed(name)` | `McpToolset(connection_params, tool_name_prefix=name)` |
| トレース | `result.all_messages()` の `ToolCallPart` 等 | イベントの `get_function_calls()` / `get_function_responses()` |
| ツール一覧 | `TestModel` に渡して読み出す | `agent.canonical_tools()` の関数宣言を読む（LLM を呼ばない） |
| 使用量 | `result.usage`（質問全体の集計） | イベントごとの `usage_metadata` を自前で合計する（下記「仕様」） |
| APIエラー | `ModelHTTPError.status_code` | `google.genai.errors.APIError.code` |

## 設計方針

- CLI の使い方は `agent.py` と同じ（`question` / `--trace` / `--mcp` / `--list-tools`）。モジュール名だけ `app.agent_adk`
- `INSTRUCTIONS`、`MCP_CONFIG`、`api_error_message`、`get_api_key` は同じものを使う
- `agent.py` にあった「TestModel でテストする」前提が ADK には無い。モデル差し替えは `build_agent(model=...)` の
  引数で行い、テストでは偽の LLM（`BaseLlm` のサブクラス）を渡す
- `tool_name_prefix` は ADK 側で `f"{prefix}_{tool.name}"` と結合される（`base_toolset.py` で確認済み）ので、
  `tool_name_prefix="glossary"` で `glossary_search_glossary` になり、Pydantic AI 版と一致する見込み。
  念のため検証手順2の `--mcp --list-tools` の出力で実物を確認する
- 出力順は `agent.py` と揃える: 最終回答を先に表示し、そのあとに `--- trace ---`、ツール呼び出し、`[usage]` を表示する。
  実行中はイベントを溜めておき、回答表示後に `print_trace(events)` で出す（`summarize_usage(events)` もここで使う）
- 内部 API `tool._get_declaration()` を `--list-tools` で使っている。ADK のバージョンアップで壊れうるので、
  バージョンを固定して依存に入れる

## 変更内容

1. `pyproject.toml`（最初に行う）: `google-adk==2.14.0` を追加する。まず `uv add --dry-run "google-adk==2.14.0"` などで、
   既存の固定（`google-genai==2.31.0`、`pydantic-ai-slim[google,mcp]==2.57.0`）と依存解決が衝突しないか確かめる。
   衝突したら、内容を報告して方針を相談する（勝手にバージョンを動かさない）。
   `mcp` が推移的に入らなければ明示する。`uv.lock` も更新される
2. `app/agent_adk.py`: 初稿を検証・修正して確定する（上記「初稿に直すべき点」を含む）
3. `tests/test_agent_adk.py`: 新規。API を呼ばないテスト
   - 関数ツールが登録されている。`tests/test_agent.py` と同じ書き方にそろえ、
     `{"search_glossary", "get_lecture_summary"} <= 登録されたツール名` と「含む」で判定する
     （演習でツールを足しても壊れないようにするため。個数の固定はしない）
   - 偽の LLM で、ツール呼び出し → 結果がモデルに戻る → 最終回答、の配線が通る
   - `--list-tools` の出力にツール名が含まれる
   - MCP 設定（`load_mcp_servers`）が stdio / url それぞれで正しい接続パラメータになる
   - MCP 経由の実行（API 不要）: 偽の LLM と、実際の MCP サーバー（`app.mcp_server` を stdio で起動）を使い、
     接頭辞付きのツール名（例: `glossary_search_glossary`）の呼び出し → サーバーからの結果 →
     結果がモデルに戻る → 最終回答、までを確認する。終了後にサブプロセスが残らないことも確認する
   - `summarize_usage` が、複数回の LLM 呼び出しの合計と呼び出し回数を正しく返す
4. `README.md`: ADK 版の使い方を1行追記するか検討（要確認。講座資料なので触るなら最小限）

## 検証手順

1. `uv run python -m app.agent_adk --list-tools` でツール3つが表示される（APIキー不要）
2. `uv run python -m app.agent_adk --mcp --list-tools` で `glossary_` 付きのツールが表示される
3. `GEMINI_API_KEY` を設定して `uv run python -m app.agent_adk --trace "LoRAとは？"` を実行し、
   `search_glossary` が呼ばれて回答の最後に「第N回」が付く。`[usage]` の呼び出し回数が、
   ツール呼び出しを含めたモデル応答の数と一致する
4. 同じく `--mcp --trace "LoRAとは？"` を実行し、接頭辞付きのツールが呼ばれて結果が回答に反映される
5. `uv run pytest`（既存テストが壊れていないこと、新テストが通ること。MCP 経由の偽 LLM テストを含む）

## リスク・未確認事項

- ADK は MCP の依存（`mcp`）やモデルの `Client` 周りが Pydantic AI や `fastmcp-slim` と競合しないか未確認。
  `uv add` 時に依存解決で衝突したら、その内容を報告して方針を相談する
- 同期か非同期か: `run()`（同期）は `Runner` に定義され、`InMemoryRunner` が継承している（ADK 2.14.0 で確認済み）。
  ただし `canonical_tools()`（ツール一覧）、`session_service.create_session()`、`McpToolset.close()` がすべて非同期で、
  `runner.run()` だけ同期にしても `asyncio.run()` は消えない。さらにイベントループが複数になると、
  MCP のサブプロセスやトランスポートでループ不一致のエラーを起こしやすい。
  そのため全体を非同期（`amain` + `run_async`）に統一する

## 仕様として決めること

### 使用量（`[usage]`）の扱い

`agent.py` の `result.usage` は質問全体の集計（LLM 呼び出し回数 `requests`、入力・出力トークン）で、
既存教材もこの値でループの回数を確認する。ADK の `event.usage_metadata` はモデル応答ごとの値なので、
最後の1件だけ表示するとツール呼び出し前の消費が抜け、比較を誤る。そこで次の仕様にする。

- `[usage]` は質問全体の集計にする: 入力トークン合計、出力トークン合計（思考分を含む）、LLM 呼び出し回数、ツール呼び出し回数
- 集計は `usage_metadata` を持つイベントを数えて足す。ストリーミングの途中イベント（`partial=True`）は
  重複計上になりうるので除外する。この除外が正しいかは `--trace` の実行で、
  呼び出し回数がモデルの応答数と一致するかで確かめる
- 表示形式は Pydantic AI 版と完全に揃える: `RunUsage(input_tokens=…, output_tokens=…, requests=…, tool_calls=…)`
  （`RunUsage` という同名の dataclass を `agent_adk.py` に持つ。`total_tokens` は出さない）。
  比較対象は「質問全体の合計」で統一する
- 集計処理は `summarize_usage(events)` のような小さな関数に切り出し、テストできるようにする

### `make_quiz` の登録

- 現状: `agent.py` は `make_quiz` を登録済み（`agent.tool_plain(tools.make_quiz)`）。
  「TODO(第3回 演習): make_quiz を作り、ここで登録しよう」というコメントだけが古いまま残っている
- 演習開始時の状態（配布時）は、`tools.py` に `make_quiz` が無く、`agent.py` にも登録が無かったと想定される。
  ここは配布用の状態であり、今回の ADK 版の対象ではない
- ADK 版は現状の `agent.py` と同じ機能にそろえるため、`make_quiz` を登録する（外さない）。
  TODO コメントは演習の状態を示すものなので ADK 版には写さず、`agent.py` 側の整理は別件とする

## 決めたこと（最終確認待ち）

1. ファイル名は `app/agent_adk.py`（Google ADK）
2. `app/agent.py` と並存させる（既存のテストや教材を壊さない）
3. 非同期（`run_async` / `amain`）に統一する
4. `pyproject.toml` に `google-adk` をバージョン固定で追加する（最初に依存解決の dry-run で衝突を確認する）

上記は提案どおりで進めてよいか、実装前にもう一度確認する。

