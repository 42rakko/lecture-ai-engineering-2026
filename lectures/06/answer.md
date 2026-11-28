# 第6回 解答

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：glossary_search_glossary
- STEP 2 の [tool call]：glossary_search_glossary({'query': '3レベル'})
- STEP 3 の [tool call]：fetch_fetch({'url': 'https://modelcontextprotocol.io/'})
- STEP 4 の [tool call]：aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニング 概要', 'topics': ['general']})
- 気づき・疑問：mcpサーバーの設定(mcp_servers.json)を書き換えるだけで簡単にmcpサーバーの機能を追加でき、さらに外部で公開されているものも利用できることがわかる

## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：発展
- やったこと：Antigravity cliのmcp設定箇所を調べた
- 結果：
  -   調べた日 2026-11-28  https://antigravity.google/docs/mcp 
  - 設定ファイル
    - ~/.gemini/config/mcp_config.json  → グローバル設定のためすべてのプロジェクトに適用
    - workspace/.agents/mcp_config.json  →　該当するworkspaceのみ適用
  - コマンドで起動する書き方
    - command
  - URL で接続する書き方 
    - serverUrl(urlでも可)
- 気づき・疑問：これをつかえば、antigravityやcodexなどのエージェントがさらに便利になることが分かったが、実際に利用するとなると公開mcpサーバーを探したり信頼性を検証したりする必要がある。手軽なのは自作するのがいいと思った。
